"""
CosyVoice All-in-One Service: UI + API + MCP
"""
import os
import sys
import gc
import re
import io
import zipfile
import time
import uuid
import json
import asyncio
import threading
from pathlib import Path
from typing import Optional, Generator
from contextlib import asynccontextmanager

import torch
import torchaudio
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / "third_party/Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import load_wav

# Fun-ASR-Nano for auto transcription
_asr_model = None
_asr_lock = threading.Lock()
# 全局推理锁：同一时间只允许一个合成任务占用 CosyVoice 模型（页面/批量/OpenAI 接口共用）。
# CosyVoice 内部是 Qwen LLM 逐 token 生成，并发 forward 打进同一模型实例可能音频错乱或崩溃；
# 批量后台线程按句持锁，句与句之间会释放，页面请求仍可穿插响应。
_infer_lock = threading.Lock()

def get_asr_model():
    global _asr_model
    if _asr_model is None:
        with _asr_lock:
            if _asr_model is None:
                from funasr import AutoModel
                print("Loading Fun-ASR-Nano model...")
                # 加载到 GPU（必须配合 _asr_lock 防止预热线程与请求并发触发双实例加载）
                _asr_model = AutoModel(
                    model="FunAudioLLM/Fun-ASR-Nano-2512",
                    trust_remote_code=True,
                    remote_code="./model.py",
                    device="cuda:0",
                    disable_update=True,
                )
                # Fun-ASR 加载完会把 torch 全局默认 dtype 留在 bfloat16（见 frontend._extract_spk_embedding
                # 处作者的显式 float32 补丁注释）。此后所有未显式指定 dtype 的张量工厂调用都被污染成
                # bf16 —— whisper mel / kaldi.fbank 等 CPU 特征提取精度受损，页面"上传参考音频现场提取"
                # 路径的合成会退化成乱码（2026-09-06 排查定位）。加载完成后恢复 float32。
                torch.set_default_dtype(torch.float32)
                print("Fun-ASR-Nano loaded!")
    return _asr_model

def transcribe_audio(audio_path: str) -> str:
    """Use Fun-ASR-Nano to transcribe audio file"""
    model = get_asr_model()
    res = model.generate(
        input=[audio_path],
        cache={},
        batch_size=1,
        language="auto",
        itn=True,
    )
    return res[0]["text"].strip() if res else ""


# ---------- 上传音频统一转码 ----------
# 手机录音常见 m4a/webm/mp4，libsndfile（soundfile）无法解码 → 必须先转成 wav
def convert_to_wav(src: str, dst: str, sample_rate: int = 24000) -> bool:
    """用 ffmpeg 把任意音频转成单声道 wav（成功返回 True，ffmpeg 缺失返回 False）"""
    import subprocess
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-ar", str(sample_rate), "-ac", "1", dst],
            capture_output=True, timeout=60,
        )
        return r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0
    except FileNotFoundError:
        return False

def ensure_wav(src: str, dst: str, sample_rate: int = 24000) -> None:
    """把任意音频文件统一转成 sample_rate 单声道 wav。

    优先 ffmpeg（支持 m4a/webm/ogg 等）；ffmpeg 不可用或失败时退化为
    soundfile 直读重采样（仅支持 wav/flac/mp3 等常见格式）。
    无法解码时抛 HTTPException(400)。
    """
    if convert_to_wav(src, dst, sample_rate):
        return
    try:
        import soundfile as sf
        data, sr = sf.read(src, dtype='float32', always_2d=True)
        waveform = torch.from_numpy(data.T)
        if sr != sample_rate:
            import torchaudio.functional as F
            waveform = F.resample(waveform, sr, sample_rate)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        sf.write(dst, waveform.squeeze(0).numpy(), sample_rate, subtype='PCM_16')
    except Exception as e:
        raise HTTPException(400, f"音频解码失败（需 wav/mp3/m4a/webm 等格式，请确认文件未损坏）: {e}")


# 引擎 frontend._extract_speech_token 硬限制：参考音频超 30 秒直接 assert 崩溃。
# 崩溃发生在流式响应已开始之后（浏览器表现为 ERR_INCOMPLETE_CHUNKED_ENCODING），
# 无法转成 HTTP 错误码，必须在音频进入引擎前于各入口拦截。
PROMPT_MAX_SECONDS = 30.0

def check_prompt_duration(path: str, max_seconds: float = PROMPT_MAX_SECONDS) -> None:
    """校验参考音频时长不超过引擎上限，超限抛 HTTPException(400)"""
    try:
        import soundfile as sf
        info = sf.info(path)
        duration = info.frames / float(info.samplerate)
    except Exception as e:
        raise HTTPException(400, f"无法读取参考音频信息: {e}")
    if duration > max_seconds:
        raise HTTPException(
            400,
            f"参考音频时长 {duration:.1f} 秒，超过引擎上限 {max_seconds:.0f} 秒，"
            f"请截取 {max_seconds:.0f} 秒以内的片段后重试（推荐 3-30 秒）"
        )


# ---------- 复述参考文本检测与重试 ----------
# 参考文本泄漏关键词（命中任一即判定泄漏，需排除输入文本自身包含的情况）
LEAK_KEYWORDS = ["好好学习", "天天向上", "学习天天", "学习，天天"]

def normalize_for_check(s: str) -> str:
    """去掉标点/空格，便于子串匹配"""
    for ch in "，。！？、,.!?;；:：'\" \t\n":
        s = s.replace(ch, "")
    return s

def _text_to_fragments(text: str) -> list:
    """把文本切成 2-4 字的检测片段（去掉标点后）"""
    t = normalize_for_check(text)
    frags = []
    for n in (2, 3, 4):
        for i in range(0, max(1, len(t) - n + 1)):
            frags.append(t[i:i + n])
    return frags

def has_reference_leak(transcribed: str, input_text: str, prompt_text: str) -> bool:
    """检测转写结果是否泄漏了参考文本（prompt_text 或已知关键词）"""
    if not transcribed:
        return False
    t = normalize_for_check(transcribed)
    i = normalize_for_check(input_text)
    # 若整个转写基本等于输入文本（无前缀/后缀），不算泄漏
    if t == i:
        return False
    # 1) 硬编码关键词（覆盖常见参考文本）
    for kw in LEAK_KEYWORDS:
        nkw = normalize_for_check(kw)
        if nkw in t and nkw not in i:
            return True
    # 2) 通用检测：prompt_text 中 ≥3 字的片段出现在转写里、但不在输入文本里 → 泄漏
    #    （排除太短的片段避免误伤，如单个"你好"）
    if prompt_text:
        for frag in _text_to_fragments(prompt_text):
            if len(frag) >= 3 and frag in t and frag not in i:
                return True
    return False


def check_tts_quality(transcribed: str, input_text: str, prompt_text: str) -> str:
    """增强版音频质量检测，返回 'ok' / 'leak' / 'incomplete' / 'garbage'。

    在参考文本泄漏检测的基础上，增加：
    - 内容完整性：转写须包含输入文本的 ≥2 字核心片段，否则视为丢字（incomplete）
    - 内容错乱：转写与输入完全不匹配且疑似复读/乱码（garbage）
    - 语气词容忍：只丢了"好的/嗯"等低信息量前缀但主体完整 → 仍判 ok（对话场景优先响应速度）
    """
    if not transcribed:
        return "garbage"
    t = normalize_for_check(transcribed)
    i = normalize_for_check(input_text)
    if not i:
        return "ok" if t else "garbage"

    # 0) 参考文本泄漏优先检查（用户核心诉求：绝不带"好好学习天天向上"）
    if has_reference_leak(transcribed, input_text, prompt_text):
        return "leak"

    # 0.5) 复读检测：转写明显比输入长（>1.5倍）说明内容重复了两遍以上
    #      （如"我现在就帮你查询我现在就帮你查询"），人耳听感极差 → 重试
    if len(t) > max(len(i), 4) * 1.5:
        return "garbage"

    # 1) 内容完整性：提取输入文本的核心片段（跳过低信息量的语气词/客套词）
    LOW_INFO = {"好的", "好", "嗯", "恩", "呃", "啊", "哦", "是的", "对", "对的",
                "行", "好的吧", "嗯嗯", "你好", "您好", "喂"}
    # 输入去语气词后的主体
    i_body = i
    for w in sorted(LOW_INFO, key=len, reverse=True):
        i_body = i_body.replace(w, "")
    i_body = i_body.strip()
    if len(i_body) >= 2:
        # 主体 ≥2 字：主体片段必须出现在转写中（核心完整性检查）
        frags = [i_body[j:j+2] for j in range(0, max(1, len(i_body) - 1))]
        if not any(f in t for f in frags):
            # 转写里完全没有输入主体 → 错乱或复读
            if any(f in t for f in _text_to_fragments(prompt_text) if len(f) >= 2):
                return "leak"
            return "garbage"

    # 2) 完整转写覆盖输入主体的大多数 2 字片段（丢字检测）
    #    只丢了语气词前缀（"好的/嗯"等）时覆盖率会稍低，但主体完整 → 判 ok
    if len(i) >= 4:
        i_frags = [i[j:j+2] for j in range(0, len(i) - 1)]
        hit = sum(1 for f in i_frags if f in t)
        ratio = hit / len(i_frags)
        # 覆盖率 < 60% 视为明显丢字/错乱（排除只丢了"好的"等单语气词的情形——那由上面的主体检查兜底）
        if ratio < 0.6:
            # 但若主体核心 2 字片段已全部覆盖（只缺语气词），仍可接受
            if len(i_body) >= 2:
                body_frags = [i_body[j:j+2] for j in range(0, max(1, len(i_body) - 1))]
                if all(f in t for f in body_frags):
                    return "ok"
            return "garbage" if ratio < 0.4 else "incomplete"
    return "ok"

# Directories
INPUT_DIR = Path(os.getenv("INPUT_DIR", "/data/input"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/data/output"))
VOICES_DIR = Path(os.getenv("VOICES_DIR", "/data/voices"))
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VOICES_DIR.mkdir(parents=True, exist_ok=True)

# Voice Manager - 管理自定义音色
class VoiceManager:
    def __init__(self, voices_dir: Path):
        self.voices_dir = voices_dir
        self.index_file = voices_dir / "voices.json"
        self.voices = self._load_index()
    
    def _load_index(self) -> dict:
        if self.index_file.exists():
            voices = json.loads(self.index_file.read_text())
        else:
            voices = {}
        # 兼容旧版（Docker 时代）写入的绝对路径（/data/voices/...）：
        # 统一按 voice_id 重算为本地路径，避免路径失效
        for vid, v in list(voices.items()):
            local = self.voices_dir / vid / "prompt.wav"
            if local.exists():
                v["audio_path"] = str(local)
        return voices
    
    def _save_index(self):
        self.index_file.write_text(json.dumps(self.voices, ensure_ascii=False, indent=2))
    
    def create(self, name: str, text: str, audio_data: bytes) -> str:
        voice_id = uuid.uuid4().hex[:12]
        voice_dir = self.voices_dir / voice_id
        voice_dir.mkdir(exist_ok=True)
        
        audio_path = voice_dir / "prompt.wav"
        audio_path.write_bytes(audio_data)
        
        self.voices[voice_id] = {
            "id": voice_id,
            "name": name,
            "text": text,
            "audio_path": str(audio_path),
            "created_at": int(time.time())
        }
        self._save_index()
        return voice_id
    
    def get(self, voice_id: str) -> Optional[dict]:
        return self.voices.get(voice_id)
    
    def list_all(self) -> list:
        return [{"id": v["id"], "name": v["name"], "text": v["text"], "created_at": v["created_at"]} 
                for v in self.voices.values()]
    
    def delete(self, voice_id: str) -> bool:
        if voice_id not in self.voices:
            return False
        voice_dir = self.voices_dir / voice_id
        if voice_dir.exists():
            import shutil
            shutil.rmtree(voice_dir)
        del self.voices[voice_id]
        self._save_index()
        return True

    def update(self, voice_id: str, name: str = None, text: str = None) -> bool:
        if voice_id not in self.voices:
            return False
        if name is not None:
            self.voices[voice_id]["name"] = name
        if text is not None:
            self.voices[voice_id]["text"] = text
        self._save_index()
        return True

voice_manager = VoiceManager(VOICES_DIR)

def _engine_prompt_cache_size(model) -> int:
    """读取 CosyVoice 引擎 frontend 的 prompt 特征缓存条数（兼容官方原版引擎）。

    作者本机给引擎 cosyvoice/cli/frontend.py 打过补丁（增加 prompt_cache 属性），
    官方原版引擎没有该属性；统一用 getattr 读取，缺失按 0 处理，避免启动/状态查询报错。
    """
    if model is None:
        return 0
    fc = getattr(model.frontend, "prompt_cache", None)
    return len(fc) if isinstance(fc, dict) else 0


# GPU Manager - 禁用自动卸载，启动时预热
class GPUManager:
    def __init__(self):
        self.model = None
        self.model_dir = None
        self.lock = threading.Lock()
        self.prompt_cache = {}  # 缓存 prompt 特征
        
    def get_model(self, model_dir: str = None):
        with self.lock:
            if model_dir is None:
                model_dir = os.getenv("MODEL_DIR", "pretrained_models/Fun-CosyVoice3-0.5B")
            if self.model is None or self.model_dir != model_dir:
                self._load_model(model_dir)
            return self.model
    
    def _load_model(self, model_dir: str):
        if self.model is not None:
            self.offload()
        print(f"Loading model from {model_dir}...")
        self.model = AutoModel(model_dir=model_dir)
        self.model_dir = model_dir
        print(f"Model loaded successfully!")
    
    def preload(self):
        """启动时预热模型和所有音色的 embedding"""
        print("Preloading model...")
        model = self.get_model()
        print("Model preloaded!")
        
        # 预热所有已保存音色的 embedding
        voices = voice_manager.list_all()
        if voices:
            print(f"Preloading {len(voices)} voice embeddings...")
            for v in voices:
                voice = voice_manager.get(v["id"])
                if voice and os.path.exists(voice["audio_path"]):
                    try:
                        # 调用一次 frontend_zero_shot 触发缓存
                        model.frontend.frontend_zero_shot(
                            "预热", voice["text"], voice["audio_path"], 
                            24000, ""
                        )
                        print(f"  ✓ Cached: {v['name']} ({v['id']})")
                    except Exception as e:
                        print(f"  ✗ Failed: {v['name']} - {e}")
            print(f"Voice embeddings cached: {_engine_prompt_cache_size(model)}")
        
        print("Model preloaded and ready!")
    
    def offload(self):
        """手动卸载模型"""
        if self.model:
            del self.model
            self.model = None
            self.model_dir = None
            self.prompt_cache.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("GPU memory released")
    
    def get_prompt_cache(self, voice_id: str):
        """获取缓存的 prompt 特征"""
        return self.prompt_cache.get(voice_id)
    
    def set_prompt_cache(self, voice_id: str, cache_data: dict):
        """缓存 prompt 特征"""
        self.prompt_cache[voice_id] = cache_data
    
    def status(self) -> dict:
        gpu_info = {"available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            gpu_info.update({
                "device": torch.cuda.get_device_name(0),
                "memory_used": f"{torch.cuda.memory_allocated()/1024**3:.2f} GB",
                "memory_total": f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB",
            })
        # 获取真实的 frontend 缓存数量
        cache_size = _engine_prompt_cache_size(self.model)
        return {
            "model_loaded": self.model is not None,
            "model_dir": self.model_dir,
            "gpu": gpu_info,
            "prompt_cache_size": cache_size
        }

gpu_manager = GPUManager()

# FastAPI App - 启动时预热模型
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时预热模型
    gpu_manager.preload()
    # Fun-ASR 必须在开始对外服务前同步加载（2026-09-06 由后台线程改为同步）：
    # 其加载过程会把 torch 全局默认 dtype 翻成 bfloat16（见 get_asr_model 内注释），
    # 若放后台线程，加载窗口期内到达的合成请求会拿到被 bf16 污染的现场提取特征，
    # 导致页面上传参考音频合成乱码。同步加载完成后恢复 float32，竞态窗口归零。
    # 加载失败不阻断启动：转写/质量检测功能退化，合成不受影响（走惰性加载）。
    try:
        get_asr_model()
    except Exception as e:
        print(f"[ASR] Warmup failed (will lazy-load): {e}", flush=True)
    yield
    # 关闭时不自动卸载（保持模型在显存中）

app = FastAPI(
    title="CosyVoice API",
    description="""
## CosyVoice Text-to-Speech API

基于大语言模型的语音合成服务，支持：
- **零样本克隆** (zero_shot): 使用3-30秒参考音频克隆任意音色
- **跨语种克隆** (cross_lingual): 跨语言语音合成
- **指令控制** (instruct): 方言、情感、语速等控制
- **预训练音色** (sft): 使用内置音色

### 支持语言
中文、英文、日语、韩语、德语、西班牙语、法语、意大利语、俄语，以及18+种中文方言
    """,
    version="3.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 全局异常日志：任何未捕获异常都写入 runtime\server_errors.log（含完整堆栈），
# 方便在输出被窗口吞掉时排查 500
import traceback as _tb
from starlette.responses import Response as _StarletteResponse

@app.middleware("http")
async def log_500_errors(request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        try:
            err_path = ROOT_DIR.parent / "runtime" / "server_errors.log"
            err_path.parent.mkdir(parents=True, exist_ok=True)
            with open(err_path, "a", encoding="utf-8") as f:
                f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} | {request.method} {request.url.path} =====\n")
                f.write(_tb.format_exc())
                f.write("\n")
        except Exception:
            pass
        raise

# Models
class TTSRequest(BaseModel):
    text: str
    mode: str = "zero_shot"  # zero_shot, cross_lingual, instruct, sft
    prompt_text: Optional[str] = ""
    instruct_text: Optional[str] = ""
    spk_id: Optional[str] = ""
    speed: float = 1.0
    stream: bool = False

class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: float = 0
    output_file: Optional[str] = None
    error: Optional[str] = None

tasks = {}

# ---------- 长文本分段合成（按句号逐句合成） ----------
batch_tasks = {}  # task_id -> {status,total,done,failed,sentences:[...],error}

def split_text_to_sentences(text: str, max_len: int = 100) -> list:
    """把长文本按句末标点/换行切成一句一句，保留标点；超长句按逗号或长度兜底切分。

    - 句末标点：。！？；!?;
    - 丢弃纯标点片段（如对话文本中单独成行的引号 '"'），避免合成失败
    - 超长句（无标点的长段）按逗号二次切分，仍超长再按 max_len 硬切，避免单句过长导致合成差/慢
    """
    def _has_content(s: str) -> bool:
        """是否含实际内容（汉字/字母/数字），纯标点/引号视为无内容"""
        return bool(re.search(r'[\u4e00-\u9fff\u3400-\u4dbfA-Za-z0-9]', s))

    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r'(?<=[。！？；!?;])\s*|\n+', text)
    sentences = []
    for p in parts:
        p = p.strip()
        if not p or not _has_content(p):
            continue
        if len(p) <= max_len:
            sentences.append(p)
            continue
        # 超长：先按逗号/分号/顿号二次切分
        sub = re.split(r'(?<=[，,；;、])', p)
        buf = ""
        for s in sub:
            if len(buf) + len(s) <= max_len:
                buf += s
            else:
                if buf.strip() and _has_content(buf):
                    sentences.append(buf.strip())
                # 硬切兜底
                while len(s) > max_len:
                    sentences.append(s[:max_len])
                    s = s[max_len:]
                buf = s
        if buf.strip() and _has_content(buf):
            sentences.append(buf.strip())
    return sentences


def _process_batch(task_id: str, sentences: list, mode: str, voice_id: str,
                   prompt_text: str, prompt_path: str, speed: float, instruct_text: str):
    """后台线程：逐句合成（串行，GPU 一次跑一个任务最稳）"""
    task = batch_tasks[task_id]
    task["status"] = "processing"
    try:
        model = gpu_manager.get_model()
        # 确定参考音频与参考文本
        prompt_audio = None
        full_prompt_text = prompt_text or ""
        if voice_id:
            voice = voice_manager.get(voice_id)
            if voice and os.path.exists(voice["audio_path"]):
                prompt_audio = voice["audio_path"]
                if not full_prompt_text:
                    full_prompt_text = voice.get("text", "") or ""
        elif prompt_path:
            prompt_audio = prompt_path
            if not full_prompt_text:
                try:
                    full_prompt_text = transcribe_audio(prompt_path)
                except Exception:
                    full_prompt_text = ""

        if prompt_audio:
            # 参考音频超 30s 会让引擎 assert 崩溃，逐句合成前整体拦截
            check_prompt_duration(prompt_audio)

        for idx, sentence in enumerate(sentences):
            st = task["sentences"][idx]
            st["status"] = "processing"
            filename = f"batch_{task_id}_{idx:04d}.wav"
            try:
                if mode == "cross_lingual":
                    if not prompt_audio:
                        raise ValueError("cross_lingual 模式需要参考音频（选择音色或上传音频）")
                    output = model.inference_cross_lingual(sentence, prompt_audio, stream=False, speed=speed)
                elif mode == "instruct":
                    if not prompt_audio:
                        raise ValueError("instruct 模式需要参考音频（选择音色或上传音频）")
                    if hasattr(model, 'inference_instruct2'):
                        output = model.inference_instruct2(sentence, instruct_text, prompt_audio, stream=False, speed=speed)
                    else:
                        output = model.inference_instruct(sentence, "", instruct_text, stream=False, speed=speed)
                else:  # zero_shot（默认，音色最像）
                    if not prompt_audio:
                        raise ValueError("zero_shot 模式需要参考音频（选择音色或上传音频）")
                    base = full_prompt_text.strip() or "你好，这是一段语音示例。"
                    ptext = f"You are a helpful assistant.<|endofprompt|>{base[:50]}"
                    output = model.inference_zero_shot(sentence, ptext, prompt_audio, stream=False, speed=speed)

                # 全局推理锁：批量逐句合成与页面/OpenAI 请求共用一把锁，避免并发打进同一模型
                with _infer_lock:
                    speeches = [chunk['tts_speech'] for chunk in output]
                if not speeches:
                    raise ValueError("模型未产出音频")
                full_speech = torch.cat(speeches, dim=1)
                save_audio(full_speech, model.sample_rate, filename)
                st.update(status="done", filename=filename)
            except Exception as e:
                st.update(status="failed", error=str(e))
                task["failed"] += 1
            task["done"] += 1
            print(f"[BATCH] {task_id} {task['done']}/{task['total']} {sentence[:20]!r} -> {st['status']}", flush=True)

        task["status"] = "completed"
    except Exception as e:
        task["status"] = "failed"
        task["error"] = str(e)
        print(f"[BATCH] {task_id} task failed: {e}", flush=True)
    finally:
        if prompt_path and os.path.exists(prompt_path):
            try:
                os.unlink(prompt_path)
            except Exception:
                pass

# Helper
def save_audio(speech: torch.Tensor, sample_rate: int, filename: str) -> str:
    output_path = OUTPUT_DIR / filename
    # Windows 原生环境：torchaudio 2.11 的 save 依赖 torchcodec，改用 soundfile 写 wav
    import soundfile as sf
    sf.write(str(output_path), speech.squeeze(0).detach().cpu().numpy(), sample_rate, subtype='PCM_16')
    return str(output_path)

def generate_audio_stream(model_output, sample_rate: int, cleanup_path: str = None):
    """Generate PCM audio stream with fade-in and DC offset removal"""
    is_first_chunk = True
    dc_offset = 0.0
    alpha = 0.001  # DC offset sliding average coefficient

    _infer_lock.acquire()
    try:
        for chunk in model_output:
            wav_chunk = chunk['tts_speech'].numpy().flatten()
            
            # Remove DC offset using sliding average
            chunk_mean = np.mean(wav_chunk)
            dc_offset = dc_offset * (1 - alpha) + chunk_mean * alpha
            wav_chunk = wav_chunk - dc_offset
            
            # Apply fade-in to first chunk
            if is_first_chunk:
                fade_len = min(2048, len(wav_chunk))
                fade = np.linspace(0, 1, fade_len)
                wav_chunk[:fade_len] *= fade
                is_first_chunk = False
            
            # Convert to int16 PCM
            audio = (wav_chunk * 32767).astype(np.int16).tobytes()
            yield audio
    finally:
        _infer_lock.release()
        # Cleanup temp file after streaming completes
        if cleanup_path and Path(cleanup_path).exists():
            Path(cleanup_path).unlink()

# ============== OpenAI-Compatible API ==============

class SpeechRequest(BaseModel):
    model: str = "cosyvoice-v3"
    input: str
    voice: str = "default"
    response_format: str = "wav"  # wav, pcm
    speed: float = 1.0
    instruct: Optional[str] = None  # 指令文本（方言、情感等）
    prompt_text: Optional[str] = None  # 参考文本（零样本克隆用）
    mode: Optional[str] = None  # 合成模式覆盖：zero_shot / cross_lingual / auto（默认 auto）
    skip_check: bool = False  # 跳过检测重试（测试/对比用）

@app.middleware("http")
async def dump_request_body(request, call_next):
    """记录每个 /v1/audio/speech 请求的原始 body，排查 Open WebUI 链路差异"""
    if request.url.path == "/v1/audio/speech":
        body = await request.body()
        print(f"[TTS-RAW] {request.client.host}:{request.client.port} body={body.decode('utf-8', errors='replace')}", flush=True)
    return await call_next(request)

@app.post("/v1/audio/speech")
def openai_speech(request: SpeechRequest):
    """OpenAI-compatible TTS API"""
    model = gpu_manager.get_model()
    # 调试日志：记录完整请求，对比 Open WebUI 链路与直连的差异
    print(f"[TTS-DEBUG] input={request.input!r} voice={request.voice!r} prompt_text={request.prompt_text!r} speed={request.speed} response_format={request.response_format} instruct={request.instruct!r}", flush=True)
    
    # 检查是否是自定义音色
    custom_voice = voice_manager.get(request.voice)
    
    if custom_voice:
        # 使用自定义音色
        prompt_audio = custom_voice["audio_path"]
        # 存量音色可能超 30s（引擎 assert 崩溃点），进入推理前拦截
        check_prompt_duration(prompt_audio)
        prompt_text = request.prompt_text or custom_voice.get("text", "") or ""
        if prompt_text.strip():
            # 完整参考文本保留用于音色克隆（不截断，避免破坏特征提取）
            full_prompt_text = prompt_text.strip()
        else:
            # 参考文字为空时，尝试自动转写；转写也失败则用默认文字
            try:
                full_prompt_text = transcribe_audio(prompt_audio)
                if not full_prompt_text.strip():
                    full_prompt_text = "你好，这是一段语音示例。"
            except Exception:
                full_prompt_text = "你好，这是一段语音示例。"

        # 参考文本保留完整（不截断，避免破坏特征提取）——zero_shot 音色特征只来自参考音频
        # 检测/重试时用于判断"是否复述了参考文本"
        full_prompt_text = (request.prompt_text or custom_voice.get("text", "") or "").strip()

        # === 参考文本格式化（根治复述：官方 <|endofprompt|> 分隔符） ===
        # 实测（大量实验）：zero_shot 会把 prompt_text 当作"待朗读内容"而复述（概率性，
        # "好的/嗯/早上好"开头高发），只能靠重试碰运气，慢且不稳。
        # CosyVoice 官方 README 推荐格式：'You are a helpful assistant.<|endofprompt|>参考文本'
        # ——<|endofprompt|> 是 LLM 系统提示词结束标记，模型知道"后面是身份描述，不是要读的内容"。
        # 实测：历史顽固案例"好的，我现在就帮你查询"从必泄漏20s+ → attempt=1 OK 1.4-2s，
        #       音色相似度 0.76（vs 原始格式 0.84，略降但远好于 cross_lingual 0.74）。
        # 参考文本太长会挤占上下文，取前 50 字足够（音色特征来自音频，文本只做身份引导）。
        prompt_text = f"You are a helpful assistant.<|endofprompt|>{full_prompt_text[:50]}"

        # === 高危文本检测（备用路由，一般不再需要） ===
        # endofprompt 已根治复述，auto 模式全部走 zero_shot。此判定保留用于极端兜底。
        import re as _re
        input_stripped = request.input.strip()
        high_risk = (
            len(input_stripped) <= 20
            and _re.match(r'^(好的|好|嗯|恩|哦|啊|行|对|是的|呃|嗯嗯|好的吧|哈哈哈|哈哈|哎|诶|早上好|早安|嗨|喂|hello|hi)', input_stripped, _re.IGNORECASE) is not None
        )
        if high_risk:
            print(f"[TTS-ROUTE] high-risk input {input_stripped!r} (备用路由)", flush=True)

        # === 合成模式选择 ===
        # auto（默认）：zero_shot + endofprompt（根治复述，音色 0.76，快）
        #   泄漏/乱码重试：zero_shot（轮换 prompt）↔ cross_lingual 交替，最多 3 次
        # mode=zero_shot：强制 zero_shot
        # mode=cross_lingual：强制 cross_lingual
        if request.instruct:
            # instruct 模式
            if hasattr(model, 'inference_instruct2'):
                output = model.inference_instruct2(
                    request.input, request.instruct, prompt_audio,
                    stream=(request.response_format == "pcm"), speed=request.speed
                )
            else:
                output = model.inference_zero_shot(
                    request.input, prompt_text, prompt_audio,
                    stream=(request.response_format == "pcm"), speed=request.speed
                )
        elif request.mode == "cross_lingual":
            # 显式指定 cross_lingual（用于对比测试/需要内容优先的场景）
            output = model.inference_cross_lingual(
                request.input, prompt_audio,
                stream=(request.response_format == "pcm"), speed=request.speed
            )
        elif request.mode == "zero_shot":
            # 显式指定 zero_shot
            output = model.inference_zero_shot(
                request.input, prompt_text, prompt_audio,
                stream=(request.response_format == "pcm"), speed=request.speed
            )
        elif high_risk or request.mode == "zero_shot":
            # auto + 高危文本 或 显式 zero_shot：zero_shot + endofprompt（根治复述）
            # 高危文本不再降级 cross_lingual——endofprompt 已解决复述，统一走 zero_shot 保音色
            output = model.inference_zero_shot(
                request.input, prompt_text, prompt_audio,
                stream=(request.response_format == "pcm"), speed=request.speed
            )
        else:
            # auto + 普通文本：zero_shot + endofprompt（音色最准），泄漏由检测重试兜底
            output = model.inference_zero_shot(
                request.input, prompt_text, prompt_audio,
                stream=(request.response_format == "pcm"), speed=request.speed
            )
    else:
        # 使用预训练音色（如果有）
        available_spks = model.list_available_spks()
        if request.voice in available_spks:
            output = model.inference_sft(
                request.input, request.voice,
                stream=(request.response_format == "pcm"), speed=request.speed
            )
        else:
            raise HTTPException(400, f"Voice '{request.voice}' not found. Use /v1/voices to list available voices or create custom voice via /v1/voices/create")
    
    if request.response_format == "pcm":
        return StreamingResponse(
            generate_audio_stream(output, model.sample_rate),
            media_type="audio/pcm",
            headers={"X-Sample-Rate": str(model.sample_rate)}
        )
    
    # 收集所有 chunks 并返回 WAV（带参考文本泄漏检测与重试）
    # 策略：zero_shot 优先（音色最准 0.84）→ 泄漏重试 1 次（轮换 prompt）→
    #       仍泄漏则第 3 次直接 cross_lingual（内容优先兜底，音色 0.74 但保证干净）
    # 最多 3 次，避免无限重试拖慢响应（用户核心诉求：快 + 干净 + 音色准）
    max_attempts = 3
    last_quality = "ok"
    last_transcribed = ""
    for attempt in range(1, max_attempts + 1):
        with _infer_lock:
            speeches = [chunk['tts_speech'] for chunk in output]
        if not speeches:
            # 模型未产出任何音频（常见原因：输入为空/乱码/无法分词，如客户端编码错误）
            raise HTTPException(422, f"合成失败：模型未产出音频。请检查输入文本是否为空或包含乱码（当前 input={request.input!r}）")
        full_speech = torch.cat(speeches, dim=1)
        filename = f"speech_{uuid.uuid4().hex[:8]}.wav"
        output_path = save_audio(full_speech, model.sample_rate, filename)

        # 只在自定义音色 + zero_shot 场景做检测（此时才存在参考文本）
        # === 检测与兜底（不再依赖重复合成） ===
        # cross_lingual 机制上不复述参考文本，因此第一次合成即可直接返回；
        # 检测仅作安全兜底（防偶发 garbage/incomplete），命中才降级 zero_shot 重试。
        do_check = bool(custom_voice) and not request.instruct and not request.skip_check
        if not do_check:
            return FileResponse(output_path, media_type="audio/wav", filename=filename)

        try:
            transcribed = transcribe_audio(str(output_path))
            quality = check_tts_quality(
                transcribed, request.input,
                full_prompt_text if custom_voice else (request.prompt_text or "")
            )
            last_quality = quality
            last_transcribed = transcribed
            if quality == "ok":
                print(f"[TTS-CHECK] attempt={attempt} OK transcribed={transcribed!r}", flush=True)
                return FileResponse(output_path, media_type="audio/wav", filename=filename)
            print(f"[TTS-CHECK] attempt={attempt} {quality.upper()} transcribed={transcribed!r} -> retry", flush=True)
        except Exception as e:
            # 检测失败不阻塞正常返回
            print(f"[TTS-CHECK] attempt={attempt} check failed: {e}, returning as-is", flush=True)
            return FileResponse(output_path, media_type="audio/wav", filename=filename)

        # 泄漏/内容问题 → 重试。策略：zero_shot + endofprompt 为主（音色准、不复述），
        # 重试时保持完整 endofprompt 文本 + 扰动词前缀（打破复述惯性，不破坏音色特征），
        # 仅最后一次才 cross_lingual 兜底（内容优先），最多 3 次。
        if attempt < max_attempts:
            # 重试：保持完整 endofprompt 文本 + 轮换扰动词前缀（嗯/呃/哦/啊）
            # 注意：不截断参考文本——实测截断反而加剧 GARBAGE
            if custom_voice and not request.instruct:
                jitter = ["嗯，", "呃，", "哦，", "啊，"][(attempt - 1) % 4]
                retry_prompt_text = f"You are a helpful assistant.<|endofprompt|>{jitter}{full_prompt_text[:50]}"
                if retry_prompt_text != prompt_text:
                    print(f"[TTS-CHECK] retry prompt_text(jitter)={retry_prompt_text!r}", flush=True)
                prompt_text = retry_prompt_text
            if request.instruct:
                if hasattr(model, 'inference_instruct2'):
                    output = model.inference_instruct2(
                        request.input, request.instruct, prompt_audio,
                        stream=False, speed=request.speed
                    )
                else:
                    output = model.inference_zero_shot(
                        request.input, prompt_text, prompt_audio,
                        stream=False, speed=request.speed
                    )
            elif attempt == max_attempts - 1:
                # 最后一次重试：cross_lingual 兜底（内容优先，音色略降但保证干净）
                print(f"[TTS-CHECK] attempt={attempt} use cross_lingual (最后兜底)", flush=True)
                output = model.inference_cross_lingual(
                    request.input, prompt_audio,
                    stream=False, speed=request.speed
                )
            else:
                output = model.inference_zero_shot(
                    request.input, prompt_text, prompt_audio,
                    stream=False, speed=request.speed
                )
        else:
            # 已达最大重试次数，返回最后一次结果（不理想但可用）
            print(f"[TTS-CHECK] max attempts reached, last_quality={last_quality} transcribed={last_transcribed!r}, returning last result", flush=True)
            return FileResponse(output_path, media_type="audio/wav", filename=filename)

@app.post("/v1/voices/create")
def create_voice(
    audio: UploadFile = File(...),
    name: str = Form(...),
    text: str = Form("")
):
    """创建自定义音色"""
    content = audio.file.read()
    
    # 统一转成 24k 单声道 wav 再保存（m4a/webm 需 ffmpeg 解码；确保后续推理/转写都能用）
    raw_path = INPUT_DIR / f"voice_raw_{uuid.uuid4().hex}.bin"
    raw_path.write_bytes(content)
    temp_path = INPUT_DIR / f"temp_{uuid.uuid4().hex}.wav"
    
    try:
        ensure_wav(str(raw_path), str(temp_path))
        check_prompt_duration(str(temp_path))
        wav_bytes = temp_path.read_bytes()
        # 如果没有提供文本，使用 Fun-ASR 转写
        if not text:
            text = transcribe_audio(str(temp_path))
        
        voice_id = voice_manager.create(name, text, wav_bytes)
        return {
            "success": True,
            "voice_id": voice_id,
            "name": name,
            "text": text,
            "message": f"音色创建成功，使用 voice='{voice_id}' 调用 /v1/audio/speech"
        }
    finally:
        if raw_path.exists():
            raw_path.unlink()
        if temp_path.exists():
            temp_path.unlink()

@app.get("/v1/voices/custom")
async def list_custom_voices():
    """列出所有自定义音色"""
    return {"voices": voice_manager.list_all()}

@app.get("/v1/voices/{voice_id}")
async def get_voice(voice_id: str):
    """获取音色详情"""
    voice = voice_manager.get(voice_id)
    if not voice:
        raise HTTPException(404, "Voice not found")
    return voice

@app.delete("/v1/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """删除自定义音色"""
    if voice_manager.delete(voice_id):
        return {"success": True, "message": "Voice deleted"}
    raise HTTPException(404, "Voice not found")

@app.get("/v1/voices")
async def list_voices():
    """列出所有可用音色（预训练 + 自定义）"""
    model = gpu_manager.get_model()
    preset_voices = model.list_available_spks()
    custom_voices = voice_manager.list_all()
    return {
        "preset_voices": preset_voices,
        "custom_voices": custom_voices
    }

@app.post("/v1/voices/{voice_id}/retranscribe")
def retranscribe_voice(voice_id: str):
    """用 Fun-ASR 重新转写参考音频，修正音色的 text（修复注册文本错误导致的克隆不准）"""
    voice = voice_manager.get(voice_id)
    if not voice:
        raise HTTPException(404, "Voice not found")
    try:
        new_text = transcribe_audio(voice["audio_path"]).strip()
        if not new_text:
            return {"success": False, "message": "转写为空，请检查参考音频", "text": ""}
        voice_manager.update(voice_id, text=new_text)
        print(f"[VOICE-RETRANSCRIBE] {voice_id} text={new_text!r}", flush=True)
        return {"success": True, "voice_id": voice_id, "text": new_text,
                "message": f"已用实际音频内容更新音色文本: {new_text}"}
    except Exception as e:
        print(f"[VOICE-RETRANSCRIBE] {voice_id} failed: {e}", flush=True)
        return {"success": False, "message": f"转写失败: {e}", "text": ""}

@app.get("/v1/audio/voices")
async def openai_list_voices():
    """OpenAI 兼容的音色列表接口（供 Open WebUI 等客户端动态拉取克隆音色）

    Open WebUI 后端 get_available_voices() 对 openai 引擎会请求
    {api_base_url}/audio/voices，期望返回 {"voices": [{"id": "...", "name": "..."}]}。
    此端点将自定义克隆音色映射为该格式，使前端音色下拉框能显示克隆音色。
    """
    custom_voices = voice_manager.list_all()
    voices = [
        {"id": v["id"], "name": v["name"]}
        for v in custom_voices
    ]
    return {"voices": voices}

@app.post("/v1/audio/transcriptions")
def openai_transcriptions(audio: UploadFile = File(...)):
    """OpenAI 兼容的语音识别接口（供 api-gateway / 手机端"打电话"使用）

    接收用户语音（wav/mp3/m4a/webm 等），用 Fun-ASR-Nano 转写为文本。
    返回格式：{"text": "..."}
    """
    import subprocess
    content = audio.file.read()
    if not content:
        raise HTTPException(400, "空音频文件")

    suffix = Path(audio.filename or "").suffix.lower() or ".wav"
    raw_path = INPUT_DIR / f"stt_{uuid.uuid4().hex}{suffix}"
    wav_path = INPUT_DIR / f"stt_{uuid.uuid4().hex}.wav"
    raw_path.write_bytes(content)
    try:
        # 转成 16k 单声道 wav（Fun-ASR 输入要求；webm/opus 等容器需 ffmpeg 解码）
        ok = False
        try:
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", str(raw_path),
                 "-ar", "16000", "-ac", "1", str(wav_path)],
                capture_output=True, timeout=60,
            )
            ok = r.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 0
        except FileNotFoundError:
            ok = False
        if not ok:
            # 没有 ffmpeg 时退化为 soundfile 解码（支持 wav/flac/ogg/mp3 等常见格式）
            try:
                import soundfile as sf
                data, sr = sf.read(str(raw_path), dtype='float32', always_2d=True)
                waveform = torch.from_numpy(data.T)
                if sr != 16000:
                    import torchaudio.functional as F
                    waveform = F.resample(waveform, sr, 16000)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                sf.write(str(wav_path), waveform.squeeze(0).numpy(), 16000, subtype='PCM_16')
            except Exception as e:
                raise HTTPException(400, f"音频解码失败（需 wav/mp3/m4a/webm，ffmpeg 不可用）: {e}")

        text = transcribe_audio(str(wav_path))
        if not text:
            raise HTTPException(422, "语音识别结果为空，请重新录音")
        print(f"[STT] {audio.filename or 'audio'} -> {text!r}", flush=True)
        return {"text": text}
    finally:
        if raw_path.exists():
            raw_path.unlink()
        if wav_path.exists():
            wav_path.unlink()

@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "models": [
            {"id": "cosyvoice-v3", "name": "Fun-CosyVoice3-0.5B", "description": "最新版本，效果最好"},
            {"id": "cosyvoice-v2", "name": "CosyVoice2-0.5B", "description": "稳定版本"},
        ]
    }

# ============== Legacy API ==============

@app.get("/health")
async def health():
    return {"status": "healthy", "gpu": gpu_manager.status()}

@app.get("/api/status")
async def api_status():
    return gpu_manager.status()

@app.post("/api/offload")
async def offload_gpu():
    gpu_manager.offload()
    return {"status": "success", "message": "GPU memory released"}

@app.get("/api/speakers")
async def list_speakers():
    model = gpu_manager.get_model()
    return {"speakers": model.list_available_spks()}

@app.post("/api/tts")
def tts(
    text: str = Form(...),
    mode: str = Form("zero_shot"),
    prompt_text: str = Form(""),
    instruct_text: str = Form(""),
    spk_id: str = Form(""),
    speed: float = Form(1.0),
    stream: bool = Form(False),
    prompt_wav: Optional[UploadFile] = File(None)
):
    model = gpu_manager.get_model()
    prompt_audio = None
    
    if prompt_wav:
        content = prompt_wav.file.read()
        # 统一转成 24k 单声道 wav（手机录音常见 m4a/webm，libsndfile 无法解码）
        raw_path = INPUT_DIR / f"prompt_raw_{uuid.uuid4().hex}.bin"
        raw_path.write_bytes(content)
        temp_path = INPUT_DIR / f"prompt_{uuid.uuid4().hex}.wav"
        try:
            ensure_wav(str(raw_path), str(temp_path))
            check_prompt_duration(str(temp_path))
            prompt_audio = str(temp_path)
        finally:
            if raw_path.exists():
                raw_path.unlink()
    
    try:
        # 参数验证
        if mode == "zero_shot":
            if not prompt_audio:
                raise HTTPException(400, "zero_shot mode requires prompt_wav (reference audio)")
            # 自动识别 prompt_text
            if not prompt_text:
                print("Auto transcribing prompt audio with Fun-ASR...")
                prompt_text = transcribe_audio(prompt_audio)
                print(f"Transcribed: {prompt_text}")
        elif mode == "cross_lingual":
            if not prompt_audio:
                raise HTTPException(400, "cross_lingual mode requires prompt_wav (reference audio)")
        elif mode == "instruct":
            if not prompt_audio:
                raise HTTPException(400, "instruct mode requires prompt_wav (reference audio)")
            if not instruct_text:
                raise HTTPException(400, "instruct mode requires instruct_text")
        elif mode == "sft":
            if not spk_id:
                raise HTTPException(400, "sft mode requires spk_id (speaker ID)")
        
        if mode == "sft":
            output = model.inference_sft(text, spk_id, stream=stream, speed=speed)
        elif mode == "zero_shot":
            output = model.inference_zero_shot(text, prompt_text, prompt_audio, stream=stream, speed=speed)
        elif mode == "cross_lingual":
            output = model.inference_cross_lingual(text, prompt_audio, stream=stream, speed=speed)
        elif mode == "instruct":
            if hasattr(model, 'inference_instruct2'):
                output = model.inference_instruct2(text, instruct_text, prompt_audio, stream=stream, speed=speed)
            else:
                output = model.inference_instruct(text, spk_id, instruct_text, stream=stream, speed=speed)
        else:
            raise HTTPException(400, f"Unknown mode: {mode}")
        
        if stream:
            return StreamingResponse(
                generate_audio_stream(output, model.sample_rate, cleanup_path=prompt_audio),
                media_type="audio/pcm"
            )
        
        # 非流式：合成 + 质量检测重试（与 /v1/audio/speech 同一套兜底；此前页面接口没有检测，
        # zero_shot 概率性复述/乱码会直接返回给用户）。策略：attempt1 原样合成 → 重试加扰动词 +
        # endofprompt 格式 → 最后一次 cross_lingual 兜底（内容优先）。仅 zero_shot 且有参考文本时检测。
        prompt_base = prompt_text or ""
        do_check = bool(mode == "zero_shot" and prompt_base.strip() and text.strip())
        for attempt in range(1, 4):  # max_attempts = 3
            with _infer_lock:
                speeches = [chunk['tts_speech'] for chunk in output]
            if not speeches:
                raise HTTPException(422, f"合成失败：模型未产出音频。请检查输入文本是否为空或包含乱码（当前 text={text!r}）")
            full_speech = torch.cat(speeches, dim=1)
            filename = f"tts_{uuid.uuid4().hex}.wav"
            output_path = save_audio(full_speech, model.sample_rate, filename)
            if not do_check:
                break
            try:
                transcribed = transcribe_audio(str(output_path))
                quality = check_tts_quality(transcribed, text, prompt_base)
                if quality == "ok":
                    print(f"[TTS-CHECK] /api/tts attempt={attempt} OK transcribed={transcribed!r}", flush=True)
                    break
                print(f"[TTS-CHECK] /api/tts attempt={attempt} {quality.upper()} transcribed={transcribed!r} -> retry", flush=True)
            except Exception as e:
                # 检测失败不阻塞正常返回
                print(f"[TTS-CHECK] /api/tts attempt={attempt} check failed: {e}, returning as-is", flush=True)
                break
            if attempt < 3:
                jitter = ["嗯，", "呃，", "哦，", "啊，"][(attempt - 1) % 4]
                prompt_text = f"You are a helpful assistant.<|endofprompt|>{jitter}{prompt_base[:50]}"
                if attempt == 2:
                    print("[TTS-CHECK] /api/tts 最后一次重试改用 cross_lingual 兜底", flush=True)
                    output = model.inference_cross_lingual(text, prompt_audio, stream=False, speed=speed)
                else:
                    output = model.inference_zero_shot(text, prompt_text, prompt_audio, stream=False, speed=speed)
            # attempt==3：已到最大次数，返回最后一次结果（不理想但可用）

        # Cleanup temp file for non-streaming mode
        if prompt_audio and Path(prompt_audio).exists():
            Path(prompt_audio).unlink()

        return FileResponse(output_path, media_type="audio/wav", filename=filename)
    
    except Exception as e:
        # Cleanup on error
        if prompt_audio and Path(prompt_audio).exists():
            Path(prompt_audio).unlink()
        raise

@app.post("/api/tts/async")
def tts_async(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    mode: str = Form("zero_shot"),
    prompt_text: str = Form(""),
    instruct_text: str = Form(""),
    spk_id: str = Form(""),
    speed: float = Form(1.0),
    prompt_wav: Optional[UploadFile] = File(None)
):
    task_id = uuid.uuid4().hex
    tasks[task_id] = {"status": "pending", "progress": 0}
    
    prompt_path = None
    if prompt_wav:
        content = prompt_wav.file.read()
        raw_path = INPUT_DIR / f"prompt_raw_{task_id}.bin"
        raw_path.write_bytes(content)
        prompt_path = INPUT_DIR / f"prompt_{task_id}.wav"
        try:
            ensure_wav(str(raw_path), str(prompt_path))
            check_prompt_duration(str(prompt_path))
        finally:
            if raw_path.exists():
                raw_path.unlink()
    
    def process():
        try:
            tasks[task_id]["status"] = "processing"
            model = gpu_manager.get_model()
            
            if mode == "sft":
                output = model.inference_sft(text, spk_id, stream=False, speed=speed)
            elif mode == "zero_shot":
                output = model.inference_zero_shot(text, prompt_text, str(prompt_path) if prompt_path else None, stream=False, speed=speed)
            elif mode == "cross_lingual":
                output = model.inference_cross_lingual(text, str(prompt_path) if prompt_path else None, stream=False, speed=speed)
            elif mode == "instruct":
                if hasattr(model, 'inference_instruct2'):
                    output = model.inference_instruct2(text, instruct_text, str(prompt_path) if prompt_path else None, stream=False, speed=speed)
                else:
                    output = model.inference_instruct(text, spk_id, instruct_text, stream=False, speed=speed)
            
            with _infer_lock:
                speeches = [chunk['tts_speech'] for chunk in output]
            if not speeches:
                raise ValueError(f"模型未产出音频（text={text!r}）")
            full_speech = torch.cat(speeches, dim=1)
            filename = f"tts_{task_id}.wav"
            save_audio(full_speech, model.sample_rate, filename)
            
            tasks[task_id] = {"status": "completed", "progress": 100, "output_file": filename}
        except Exception as e:
            tasks[task_id] = {"status": "failed", "error": str(e)}
        finally:
            if prompt_path and prompt_path.exists():
                prompt_path.unlink()
    
    background_tasks.add_task(process)
    return {"task_id": task_id}

@app.get("/api/task/{task_id}")
async def get_task(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")
    return tasks[task_id]

@app.get("/api/download/{filename}")
async def download(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(path), media_type="audio/wav", filename=filename)

@app.get("/favicon.ico")
async def favicon():
    """浏览器自动请求 favicon，无图标时返回 204 避免控制台 404 报错"""
    return Response(status_code=204)

# ============== 长文本分段合成 API ==============

@app.post("/api/tts/batch/split")
async def batch_split_preview(payload: dict):
    """按句号切分长文本，返回句子列表（预览用，不合成）"""
    text = (payload.get("text") or "").strip()
    max_len = int(payload.get("max_len", 100))
    sentences = split_text_to_sentences(text, max_len)
    return {
        "total": len(sentences),
        "sentences": [{"index": i, "text": s, "chars": len(s)} for i, s in enumerate(sentences)],
    }

@app.post("/api/tts/batch")
def create_batch_tts(
    text: str = Form(...),
    mode: str = Form("zero_shot"),
    voice_id: str = Form(""),
    prompt_text: str = Form(""),
    speed: float = Form(1.0),
    instruct_text: str = Form(""),
    max_len: int = Form(100),
    prompt_wav: Optional[UploadFile] = File(None),
):
    """长文本分段合成：按句号切分 → 后台逐句合成 → 每段单独下载 / 一键打包下载

    - voice_id 传已有音色，或 prompt_wav 上传参考音频
    - 返回 task_id，用 GET /api/tts/batch/{task_id} 轮询进度
    """
    sentences = split_text_to_sentences(text, max_len)
    if not sentences:
        raise HTTPException(400, "文本为空或没有可合成的句子")
    if not voice_id and not prompt_wav:
        raise HTTPException(400, "请选择音色（voice_id）或上传参考音频（prompt_wav）")

    task_id = uuid.uuid4().hex
    prompt_path = None
    if prompt_wav:
        content = prompt_wav.file.read()
        raw_path = INPUT_DIR / f"batch_raw_{task_id}.bin"
        raw_path.write_bytes(content)
        prompt_path = INPUT_DIR / f"batch_prompt_{task_id}.wav"
        try:
            ensure_wav(str(raw_path), str(prompt_path))
            check_prompt_duration(str(prompt_path))
        finally:
            if raw_path.exists():
                raw_path.unlink()

    batch_tasks[task_id] = {
        "status": "pending",
        "total": len(sentences),
        "done": 0,
        "failed": 0,
        "error": None,
        "sentences": [
            {"index": i, "text": s, "status": "pending", "filename": None, "error": None}
            for i, s in enumerate(sentences)
        ],
        "created_at": int(time.time()),
    }
    threading.Thread(
        target=_process_batch,
        args=(task_id, sentences, mode, voice_id, prompt_text, prompt_path, speed, instruct_text),
        daemon=True,
    ).start()
    return {"task_id": task_id, "total": len(sentences), "message": f"已创建分段合成任务，共 {len(sentences)} 句"}

@app.get("/api/tts/batch/{task_id}")
async def get_batch_task(task_id: str):
    """查询分段合成任务进度（含每句状态）"""
    if task_id not in batch_tasks:
        raise HTTPException(404, "Task not found")
    t = batch_tasks[task_id]
    return {
        "task_id": task_id,
        "status": t["status"],
        "total": t["total"],
        "done": t["done"],
        "failed": t["failed"],
        "error": t.get("error"),
        "sentences": [
            {
                "index": s["index"],
                "text": s["text"],
                "status": s["status"],
                "filename": s["filename"],
                "error": s["error"],
            }
            for s in t["sentences"]
        ],
    }

@app.get("/api/tts/batch/{task_id}/download")
def download_batch_zip(task_id: str):
    """一键下载全部已完成段落（打包 zip）"""
    if task_id not in batch_tasks:
        raise HTTPException(404, "Task not found")
    t = batch_tasks[task_id]
    files = []
    for s in t["sentences"]:
        if s.get("filename") and (OUTPUT_DIR / s["filename"]).exists():
            files.append(s["filename"])
    if not files:
        raise HTTPException(404, "还没有已完成的音频段落")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, fn in enumerate(files):
            zf.write(OUTPUT_DIR / fn, arcname=f"第{i+1:04d}段_{fn}")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="batch_{task_id}.zip"'},
    )

# UI
@app.get("/", response_class=HTMLResponse)
async def ui():
    return HTML_TEMPLATE

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CosyVoice - Text to Speech</title>
    <style>
        :root {
            --bg: #1a1a2e; --card: #16213e; --primary: #0f3460; --accent: #e94560;
            --text: #eee; --text-muted: #aaa; --border: #0f3460; --danger: #c0392b;
        }
        [data-theme="light"] {
            --bg: #f5f5f5; --card: #fff; --primary: #e3f2fd; --accent: #1976d2;
            --text: #333; --text-muted: #666; --border: #ddd; --danger: #e74c3c;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
        .container { max-width: 900px; margin: 0 auto; padding: 20px; }
        header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        h1 { font-size: 1.8em; background: linear-gradient(135deg, var(--accent), #ff6b6b); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .controls { display: flex; gap: 10px; }
        select, button { padding: 8px 16px; border: 1px solid var(--border); border-radius: 6px; background: var(--card); color: var(--text); cursor: pointer; }
        button:hover { background: var(--accent); color: white; }
        .card { background: var(--card); border-radius: 12px; padding: 20px; margin-bottom: 20px; border: 1px solid var(--border); }
        .card h3 { margin-bottom: 15px; color: var(--accent); }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; color: var(--text-muted); font-size: 0.9em; }
        textarea, input[type="text"], input[type="number"] { width: 100%; padding: 12px; border: 1px solid var(--border); border-radius: 8px; background: var(--bg); color: var(--text); font-size: 1em; resize: vertical; }
        textarea { min-height: 100px; }
        .row { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .btn-primary { background: var(--accent); color: white; border: none; padding: 14px 28px; font-size: 1.1em; width: 100%; }
        .btn-primary:disabled { opacity: 0.6; cursor: not-allowed; }
        audio { width: 100%; margin-top: 15px; }
        .status { padding: 10px; border-radius: 6px; background: var(--primary); margin-top: 10px; }
        .gpu-status { display: flex; justify-content: space-between; align-items: center; }
        .upload-area { border: 2px dashed var(--border); border-radius: 8px; padding: 20px; text-align: center; cursor: pointer; transition: all 0.3s; }
        .upload-area:hover { border-color: var(--accent); }
        .upload-area.dragover { background: var(--primary); }
        .hidden { display: none; }
        .tabs { display: flex; gap: 5px; margin-bottom: 15px; }
        .tab { padding: 10px 20px; border-radius: 6px 6px 0 0; cursor: pointer; background: var(--bg); }
        .tab.active { background: var(--accent); color: white; }
        .progress-bar { height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; margin-top: 10px; }
        .progress-bar-fill { height: 100%; background: var(--accent); width: 0%; transition: width 0.3s; }
        .batch-item { border: 1px solid var(--border); border-radius: 8px; padding: 10px; margin-bottom: 8px; background: var(--bg); }
        .batch-item-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; }
        .batch-audio-row { display: flex; gap: 10px; align-items: center; margin-top: 8px; flex-wrap: wrap; }
        .batch-audio-row audio { flex: 1; min-width: 200px; margin-top: 0; }
        .batch-status-pending { color: var(--text-muted); }
        .batch-status-processing { color: #f39c12; }
        .batch-status-done { color: #2ecc71; }
        .batch-status-failed { color: #e74c3c; }
        .batch-action-btn { padding: 10px 18px; border-radius: 8px; cursor: pointer; font-size: 1em; border: none; color: white; }
        .batch-action-btn:disabled { opacity: 0.6; cursor: not-allowed; }
        @media (max-width: 600px) { .row { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎙️ CosyVoice</h1>
            <div class="controls">
                <select id="lang" onchange="setLang(this.value)">
                    <option value="zh-CN">简体中文</option>
                    <option value="en">English</option>
                    <option value="zh-TW">繁體中文</option>
                    <option value="ja">日本語</option>
                </select>
                <button onclick="toggleTheme()">🌓</button>
            </div>
        </header>

        <div class="card">
            <h3 data-i18n="input">输入文本</h3>
            <div class="form-group">
                <textarea id="text" placeholder="请输入要合成的文本..." data-i18n-placeholder="textPlaceholder">收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。</textarea>
            </div>
        </div>

        <div class="card">
            <h3 data-i18n="mode">合成模式</h3>
            <div class="tabs">
                <div class="tab active" data-mode="zero_shot" data-i18n="zeroShot">零样本克隆</div>
                <div class="tab" data-mode="cross_lingual" data-i18n="crossLingual">跨语种</div>
                <div class="tab" data-mode="instruct" data-i18n="instruct">指令控制</div>
            </div>

            <div id="prompt-section">
                <!-- 音色选择 -->
                <div class="form-group">
                    <label data-i18n="voiceSelect">选择音色</label>
                    <div style="display: flex; gap: 10px; align-items: center;">
                        <select id="voice-select" style="flex: 1;" onchange="onVoiceSelect()">
                            <option value="">-- 上传新音频 --</option>
                        </select>
                        <button onclick="refreshVoices()" title="刷新列表" style="padding: 8px 12px;">🔄</button>
                        <button onclick="deleteSelectedVoice()" title="删除选中音色" style="padding: 8px 12px; background: var(--danger, #c0392b);">🗑️</button>
                    </div>
                </div>
                
                <!-- 上传新音频区域 -->
                <div id="upload-section">
                    <div class="form-group">
                        <label data-i18n="promptAudio">参考音频 (3-30秒)</label>
                        <div class="upload-area" id="upload-area">
                            <input type="file" id="prompt-file" accept="audio/*" class="hidden">
                            <p data-i18n="uploadHint">点击或拖拽上传音频文件</p>
                            <p id="file-name" style="color: var(--accent); margin-top: 10px;"></p>
                        </div>
                    </div>
                    <div class="form-group">
                        <label data-i18n="voiceName">音色名称 (可选，用于保存)</label>
                        <input type="text" id="voice-name" placeholder="如：张三的声音">
                    </div>
                    <div class="form-group" id="prompt-text-group">
                        <label data-i18n="promptText">参考文本 (留空则自动识别)</label>
                        <input type="text" id="prompt-text" placeholder="留空将使用 Fun-ASR 自动识别">
                    </div>
                    <div class="form-group">
                        <label style="display: flex; align-items: center; gap: 10px;">
                            <input type="checkbox" id="save-voice" checked> <span data-i18n="saveVoice">保存为自定义音色</span>
                        </label>
                    </div>
                </div>
            </div>

            <div id="instruct-section" class="hidden">
                <div class="form-group">
                    <label data-i18n="instructText">指令文本</label>
                    <input type="text" id="instruct-text" placeholder="用四川话说这句话">
                </div>
            </div>

            <div class="row">
                <div class="form-group">
                    <label data-i18n="speed">语速 (0.5-2.0)</label>
                    <input type="number" id="speed" value="1.0" min="0.5" max="2.0" step="0.1">
                </div>
            </div>
        </div>

        <div class="card">
            <div class="row" style="align-items: center;">
                <div class="form-group" style="margin-bottom: 0;">
                    <label style="display: flex; align-items: center; gap: 10px;">
                        <input type="checkbox" id="stream-mode" checked> <span data-i18n="streamMode">流式输出 (低延迟)</span>
                    </label>
                </div>
            </div>
            <button class="btn-primary" id="generate-btn" onclick="generate()" data-i18n="generate" style="margin-top: 15px;">生成语音</button>
            <div class="progress-bar"><div class="progress-bar-fill" id="progress"></div></div>
            <div id="timer" class="status hidden" style="text-align: center; font-size: 1.1em;"></div>
            <audio id="audio-output" controls class="hidden"></audio>
            <button id="download-btn" class="hidden" style="margin-top: 10px; padding: 10px 20px; background: var(--accent); color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 1em;">
                📥 <span data-i18n="download">下载音频</span>
            </button>
        </div>

        <div class="card" id="batch-card" style="border: 2px solid var(--accent);">
            <h3>📄 长文本分段合成（按句号逐句合成）</h3>
            <div class="form-group">
                <label data-i18n="batchText">长文本 / 文档内容（将按句号「。」自动切成一句一句，逐句合成）</label>
                <textarea id="batch-text" placeholder="在此粘贴长文本，或点击下方按钮上传 .txt 文档。合成时自动按句号/换行逐句截取，一句一段语音。"></textarea>
                <div style="display: flex; gap: 10px; align-items: center; margin-top: 10px; flex-wrap: wrap;">
                    <button onclick="document.getElementById('batch-file').click()" style="padding: 8px 14px;">📂 上传 .txt 文档</button>
                    <input type="file" id="batch-file" accept=".txt,.md,text/plain" class="hidden">
                    <span id="batch-file-name" style="color: var(--accent);"></span>
                </div>
            </div>
            <div class="form-group">
                <label style="display: flex; align-items: center; gap: 10px;">
                    <input type="checkbox" id="batch-save-voice" checked> <span data-i18n="batchSaveVoice">使用上方所选音色（已有音色或新上传的参考音频）</span>
                </label>
            </div>
            <div class="row">
                <div class="form-group">
                    <label data-i18n="speed">语速 (0.5-2.0)</label>
                    <input type="number" id="batch-speed" value="1.0" min="0.5" max="2.0" step="0.1">
                </div>
                <div class="form-group">
                    <label>分段预览</label>
                    <button id="batch-preview-btn" onclick="previewBatch()" style="width: 100%;">🔍 预览分段</button>
                </div>
            </div>
            <button class="btn-primary" id="batch-generate-btn" onclick="startBatch()" style="margin-top: 10px;">🎬 开始逐句合成</button>
            <div class="progress-bar"><div class="progress-bar-fill" id="batch-progress"></div></div>
            <div id="batch-status" class="status hidden" style="text-align: center; font-size: 1em;"></div>
            <div id="batch-result"></div>
            <button id="batch-download-all" class="hidden" style="margin-top: 12px; padding: 12px 20px; background: var(--accent); color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 1.05em; width: 100%;" onclick="downloadBatchZip()">
                📦 一键下载全部段落（zip）
            </button>
        </div>

        <div class="card">
            <div class="gpu-status">
                <span id="gpu-info" data-i18n="gpuStatus">GPU 状态: 加载中...</span>
                <button onclick="offloadGPU()" data-i18n="releaseGPU">释放显存</button>
            </div>
        </div>
    </div>

    <script>
        const i18n = {
            'zh-CN': { input: '输入文本', mode: '合成模式', zeroShot: '零样本克隆', crossLingual: '跨语种', instruct: '指令控制', promptAudio: '参考音频 (3-30秒)', promptText: '参考文本 (留空自动识别)', instructText: '指令文本', speed: '语速', generate: '生成语音', gpuStatus: 'GPU 状态', releaseGPU: '释放显存', uploadHint: '点击或拖拽上传音频', textPlaceholder: '请输入要合成的文本...', streamMode: '流式输出 (低延迟)', generating: '生成中', completed: '完成', firstChunk: '首包', totalTime: '总耗时', audioDuration: '音频', voiceSelect: '选择音色', voiceName: '音色名称', saveVoice: '保存为自定义音色', newUpload: '-- 上传新音频 --', download: '下载音频' },
            'en': { input: 'Input Text', mode: 'Synthesis Mode', zeroShot: 'Zero-shot Clone', crossLingual: 'Cross-lingual', instruct: 'Instruct', promptAudio: 'Reference Audio (3-30s)', promptText: 'Reference Text (auto if empty)', instructText: 'Instruction', speed: 'Speed', generate: 'Generate', gpuStatus: 'GPU Status', releaseGPU: 'Release GPU', uploadHint: 'Click or drag to upload', textPlaceholder: 'Enter text to synthesize...', streamMode: 'Streaming (Low Latency)', generating: 'Generating', completed: 'Completed', firstChunk: 'TTFB', totalTime: 'Total', audioDuration: 'Audio', voiceSelect: 'Select Voice', voiceName: 'Voice Name', saveVoice: 'Save as custom voice', newUpload: '-- Upload new audio --', download: 'Download' },
            'zh-TW': { input: '輸入文本', mode: '合成模式', zeroShot: '零樣本克隆', crossLingual: '跨語種', instruct: '指令控制', promptAudio: '參考音頻', promptText: '參考文本', instructText: '指令文本', speed: '語速', generate: '生成語音', gpuStatus: 'GPU 狀態', releaseGPU: '釋放顯存', uploadHint: '點擊或拖拽上傳', textPlaceholder: '請輸入要合成的文本...', streamMode: '流式輸出 (低延遲)', generating: '生成中', completed: '完成', firstChunk: '首包', totalTime: '總耗時', audioDuration: '音頻', voiceSelect: '選擇音色', voiceName: '音色名稱', saveVoice: '保存為自定義音色', newUpload: '-- 上傳新音頻 --', download: '下載音頻' },
            'ja': { input: '入力テキスト', mode: '合成モード', zeroShot: 'ゼロショット', crossLingual: '多言語', instruct: '指示制御', promptAudio: '参照音声', promptText: '参照テキスト', instructText: '指示テキスト', speed: '速度', generate: '生成', gpuStatus: 'GPU状態', releaseGPU: 'GPU解放', uploadHint: 'クリックまたはドラッグ', textPlaceholder: 'テキストを入力...', streamMode: 'ストリーミング', generating: '生成中', completed: '完了', firstChunk: '初回', totalTime: '合計', audioDuration: '音声', voiceSelect: '音声選択', voiceName: '音声名', saveVoice: 'カスタム音声として保存', newUpload: '-- 新規アップロード --', download: 'ダウンロード' }
        };
        let currentLang = 'zh-CN', currentMode = 'zero_shot', promptFile = null, selectedVoiceId = null;

        function setLang(lang) {
            currentLang = lang;
            document.querySelectorAll('[data-i18n]').forEach(el => {
                const key = el.dataset.i18n;
                if (i18n[lang][key]) el.textContent = i18n[lang][key];
            });
            document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
                const key = el.dataset.i18nPlaceholder;
                if (i18n[lang][key]) el.placeholder = i18n[lang][key];
            });
            refreshVoices();
        }

        function toggleTheme() {
            document.body.dataset.theme = document.body.dataset.theme === 'light' ? '' : 'light';
        }

        // Voice management
        async function refreshVoices() {
            try {
                const res = await fetch('/v1/voices/custom');
                const data = await res.json();
                const select = document.getElementById('voice-select');
                const t = i18n[currentLang];
                select.innerHTML = `<option value="">${t.newUpload}</option>` + 
                    data.voices.map(v => `<option value="${v.id}" data-text="${v.text}">${v.name} (${v.id})</option>`).join('');
                if (selectedVoiceId) select.value = selectedVoiceId;
                onVoiceSelect();
            } catch (e) { console.error('Failed to load voices:', e); }
        }
        
        function onVoiceSelect() {
            const select = document.getElementById('voice-select');
            selectedVoiceId = select.value;
            const uploadSection = document.getElementById('upload-section');
            uploadSection.classList.toggle('hidden', !!selectedVoiceId);
            
            // 如果选择了已有音色，填充 prompt_text
            if (selectedVoiceId) {
                const option = select.options[select.selectedIndex];
                document.getElementById('prompt-text').value = option.dataset.text || '';
            }
        }
        
        async function deleteSelectedVoice() {
            if (!selectedVoiceId) { alert('请先选择要删除的音色'); return; }
            if (!confirm('确定删除此音色？')) return;
            try {
                await fetch(`/v1/voices/${selectedVoiceId}`, { method: 'DELETE' });
                selectedVoiceId = null;
                refreshVoices();
            } catch (e) { alert('删除失败: ' + e.message); }
        }

        document.querySelectorAll('.tab').forEach(tab => {
            tab.onclick = () => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                currentMode = tab.dataset.mode;
                document.getElementById('prompt-section').classList.remove('hidden');
                document.getElementById('prompt-text-group').classList.toggle('hidden', currentMode === 'cross_lingual');
                document.getElementById('instruct-section').classList.toggle('hidden', currentMode !== 'instruct');
            };
        });

        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('prompt-file');
        uploadArea.onclick = () => fileInput.click();
        uploadArea.ondragover = e => { e.preventDefault(); uploadArea.classList.add('dragover'); };
        uploadArea.ondragleave = () => uploadArea.classList.remove('dragover');
        uploadArea.ondrop = e => { e.preventDefault(); uploadArea.classList.remove('dragover'); handleFile(e.dataTransfer.files[0]); };
        fileInput.onchange = e => handleFile(e.target.files[0]);
        function handleFile(file) { if (file) { promptFile = file; document.getElementById('file-name').textContent = file.name; } }

        // Web Audio API streaming player
        const SAMPLE_RATE = 24000;
        const MIN_BUFFER_SIZE = 12000;  // 500ms buffer at 24kHz
        const FADE_SAMPLES = 1024;
        let audioContext = null;
        let activeSources = [];
        let nextPlayTime = 0;
        
        function stopAllAudio() {
            activeSources.forEach(s => { try { s.stop(); } catch(e) {} });
            activeSources = [];
        }
        
        function applyFadeIn(arr) {
            const len = Math.min(FADE_SAMPLES, arr.length);
            for (let i = 0; i < len; i++) arr[i] *= i / len;
        }
        
        function createWavBlob(pcmData, sampleRate) {
            const numChannels = 1, bitsPerSample = 16;
            const byteRate = sampleRate * numChannels * bitsPerSample / 8;
            const blockAlign = numChannels * bitsPerSample / 8;
            const dataSize = pcmData.length;
            const buffer = new ArrayBuffer(44 + dataSize);
            const view = new DataView(buffer);
            
            const writeString = (offset, str) => { for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i)); };
            writeString(0, 'RIFF');
            view.setUint32(4, 36 + dataSize, true);
            writeString(8, 'WAVE');
            writeString(12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true);
            view.setUint16(22, numChannels, true);
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, byteRate, true);
            view.setUint16(32, blockAlign, true);
            view.setUint16(34, bitsPerSample, true);
            writeString(36, 'data');
            view.setUint32(40, dataSize, true);
            new Uint8Array(buffer, 44).set(pcmData);
            return new Blob([buffer], { type: 'audio/wav' });
        }
        
        let currentAudioBlob = null;
        document.getElementById('download-btn').onclick = () => {
            if (currentAudioBlob) {
                const url = URL.createObjectURL(currentAudioBlob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `cosyvoice_${Date.now()}.wav`;
                a.click();
                URL.revokeObjectURL(url);
            }
        };

        async function generate() {
            const btn = document.getElementById('generate-btn');
            const progress = document.getElementById('progress');
            const audio = document.getElementById('audio-output');
            const timer = document.getElementById('timer');
            const downloadBtn = document.getElementById('download-btn');
            const isStream = document.getElementById('stream-mode').checked;
            const t = i18n[currentLang];
            
            btn.disabled = true;
            progress.style.width = '10%';
            timer.classList.remove('hidden');
            audio.classList.add('hidden');
            downloadBtn.classList.add('hidden');
            stopAllAudio();
            
            let audioBlob = null;  // Store audio for download
            const startTime = Date.now();
            let timerInterval = setInterval(() => {
                const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
                timer.textContent = `⏱️ ${t.generating}... ${elapsed}s`;
            }, 100);

            try {
                progress.style.width = '30%';
                let res;
                
                // 如果选择了已有音色，使用 OpenAI 风格 API
                if (selectedVoiceId) {
                    const body = {
                        input: document.getElementById('text').value,
                        voice: selectedVoiceId,
                        response_format: isStream ? 'pcm' : 'wav',
                        speed: parseFloat(document.getElementById('speed').value)
                    };
                    if (currentMode === 'instruct') {
                        body.instruct = document.getElementById('instruct-text').value;
                    }
                    res = await fetch('/v1/audio/speech', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(body)
                    });
                } else {
                    // 上传新音频
                    if (!promptFile) throw new Error('请上传参考音频或选择已有音色');
                    
                    const formData = new FormData();
                    formData.append('text', document.getElementById('text').value);
                    formData.append('mode', currentMode);
                    formData.append('speed', document.getElementById('speed').value);
                    formData.append('stream', isStream);
                    formData.append('prompt_wav', promptFile);
                    formData.append('prompt_text', document.getElementById('prompt-text').value);
                    if (currentMode === 'instruct') formData.append('instruct_text', document.getElementById('instruct-text').value);
                    
                    res = await fetch('/api/tts', { method: 'POST', body: formData });
                    
                    // 如果勾选了保存音色，保存它
                    if (document.getElementById('save-voice').checked && res.ok) {
                        const voiceName = document.getElementById('voice-name').value || promptFile.name;
                        const saveForm = new FormData();
                        saveForm.append('audio', promptFile);
                        saveForm.append('name', voiceName);
                        saveForm.append('text', document.getElementById('prompt-text').value);
                        fetch('/v1/voices/create', { method: 'POST', body: saveForm })
                            .then(() => refreshVoices());
                    }
                }
                
                if (!res.ok) {
                    let msg = await res.text();
                    try { msg = JSON.parse(msg).detail || msg; } catch (_) {}
                    throw new Error(msg);
                }
                
                if (isStream) {
                    // Initialize Web Audio API
                    if (!audioContext) audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
                    if (audioContext.state === 'suspended') await audioContext.resume();
                    nextPlayTime = audioContext.currentTime + 0.15;
                    
                    const reader = res.body.getReader();
                    let pendingBytes = new Uint8Array(0);
                    let allPcmBytes = [];  // Collect all PCM data for download
                    let samples = [];
                    let totalSamples = 0;
                    let isFirstChunk = true;
                    let firstChunkTime = null;
                    
                    function playBuffer() {
                        if (samples.length === 0) return;
                        const float32 = new Float32Array(samples);
                        if (isFirstChunk) { applyFadeIn(float32); isFirstChunk = false; }
                        
                        const audioBuffer = audioContext.createBuffer(1, float32.length, SAMPLE_RATE);
                        audioBuffer.getChannelData(0).set(float32);
                        
                        const source = audioContext.createBufferSource();
                        source.buffer = audioBuffer;
                        source.connect(audioContext.destination);
                        activeSources.push(source);
                        source.onended = () => { const idx = activeSources.indexOf(source); if (idx > -1) activeSources.splice(idx, 1); };
                        
                        if (nextPlayTime < audioContext.currentTime - 0.1) nextPlayTime = audioContext.currentTime + 0.05;
                        source.start(nextPlayTime);
                        nextPlayTime += audioBuffer.duration;
                        totalSamples += samples.length;
                        samples = [];
                    }
                    
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) { playBuffer(); break; }
                        
                        allPcmBytes.push(value);  // Collect for download
                        
                        if (!firstChunkTime) {
                            firstChunkTime = Date.now();
                            const ttfb = ((firstChunkTime - startTime) / 1000).toFixed(2);
                            timer.textContent = `⏱️ ${t.firstChunk}: ${ttfb}s | ${t.generating}...`;
                        }
                        
                        // Combine with pending bytes
                        const combined = new Uint8Array(pendingBytes.length + value.length);
                        combined.set(pendingBytes);
                        combined.set(value, pendingBytes.length);
                        
                        // Ensure byte alignment (Int16 = 2 bytes)
                        const validLength = Math.floor(combined.length / 2) * 2;
                        const validData = combined.slice(0, validLength);
                        pendingBytes = combined.slice(validLength);
                        
                        // Convert PCM to float samples
                        const int16 = new Int16Array(validData.buffer, validData.byteOffset, validData.length / 2);
                        for (let i = 0; i < int16.length; i++) samples.push(int16[i] / 32768);
                        
                        progress.style.width = `${30 + Math.min(60, samples.length / 1000)}%`;
                        if (samples.length >= MIN_BUFFER_SIZE) playBuffer();
                    }
                    
                    // Create WAV blob for download
                    const totalLength = allPcmBytes.reduce((sum, arr) => sum + arr.length, 0);
                    const pcmData = new Uint8Array(totalLength);
                    let offset = 0;
                    for (const chunk of allPcmBytes) { pcmData.set(chunk, offset); offset += chunk.length; }
                    audioBlob = createWavBlob(pcmData, SAMPLE_RATE);
                    
                    clearInterval(timerInterval);
                    const totalTime = ((Date.now() - startTime) / 1000).toFixed(2);
                    const ttfb = firstChunkTime ? ((firstChunkTime - startTime) / 1000).toFixed(2) : '-';
                    const audioDuration = (totalSamples / 24000).toFixed(2);
                    timer.textContent = `✅ ${t.firstChunk}: ${ttfb}s | ${t.totalTime}: ${totalTime}s | ${t.audioDuration}: ${audioDuration}s`;
                    timer.style.background = 'var(--accent)';
                    downloadBtn.classList.remove('hidden');
                    progress.style.width = '100%';
                } else {
                    const blob = await res.blob();
                    audioBlob = blob;
                    audio.src = URL.createObjectURL(blob);
                    clearInterval(timerInterval);
                    const totalTime = ((Date.now() - startTime) / 1000).toFixed(2);
                    timer.textContent = `✅ ${t.totalTime}: ${totalTime}s`;
                    timer.style.background = 'var(--accent)';
                    audio.classList.remove('hidden');
                    downloadBtn.classList.remove('hidden');
                    audio.play();
                    progress.style.width = '100%';
                }
            } catch (e) {
                clearInterval(timerInterval);
                timer.textContent = '❌ Error: ' + e.message;
                timer.style.background = '#c0392b';
            } finally {
                currentAudioBlob = audioBlob;
                btn.disabled = false;
                setTimeout(() => {
                    progress.style.width = '0%';
                    timer.style.background = 'var(--primary)';
                }, 2000);
            }
        }

        async function offloadGPU() {
            await fetch('/api/offload', { method: 'POST' });
            updateStatus();
        }

        // ========== 长文本分段合成（按句号逐句合成） ==========
        let batchTaskId = null, batchPollTimer = null, batchDone = false;

        function escapeHtml(s) {
            return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
        }

        document.getElementById('batch-file').onchange = function(e) {
            const file = e.target.files[0];
            if (!file) return;
            document.getElementById('batch-file-name').textContent = `📄 ${file.name}`;
            const reader = new FileReader();
            reader.onload = ev => {
                document.getElementById('batch-text').value = ev.target.result;
                document.getElementById('batch-status').classList.remove('hidden');
                document.getElementById('batch-status').textContent = `✅ 已读取 ${file.name}（${ev.target.result.length} 字），可点「预览分段」查看切分结果`;
            };
            reader.readAsText(file, 'utf-8');
        };

        async function previewBatch() {
            const text = document.getElementById('batch-text').value.trim();
            if (!text) { alert('请先输入或上传长文本'); return; }
            const btn = document.getElementById('batch-preview-btn');
            btn.disabled = true;
            try {
                const res = await fetch('/api/tts/batch/split', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || '切分失败');
                renderBatchSentences(data.sentences.map(s => ({ ...s, status: 'pending', filename: null, error: null })));
                const st = document.getElementById('batch-status');
                st.classList.remove('hidden');
                st.textContent = `🔍 预览：共 ${data.total} 句（仅展示，点「开始逐句合成」正式生成）`;
            } catch (e) {
                alert('预览失败: ' + e.message);
            } finally {
                btn.disabled = false;
            }
        }

        function renderBatchSentences(sentences) {
            const box = document.getElementById('batch-result');
            if (!sentences.length) { box.innerHTML = ''; return; }
            box.innerHTML = sentences.map(s => `
                <div class="batch-item">
                    <div class="batch-item-head">
                        <span style="font-weight: bold; color: var(--accent);">#${s.index + 1}</span>
                        <span class="batch-status-${s.status}">${batchStatusLabel(s)}</span>
                    </div>
                    <div style="margin-top: 4px; font-size: 0.92em; color: var(--text-muted);">${escapeHtml(s.text)}</div>
                    ${s.status === 'done' && s.filename ? `
                    <div class="batch-audio-row">
                        <audio controls preload="none" src="/api/download/${s.filename}"></audio>
                        <a href="/api/download/${s.filename}" download="${s.filename}" style="color: var(--accent); white-space: nowrap;">⬇️ 下载本段</a>
                    </div>` : ''}
                    ${s.status === 'failed' && s.error ? `<div style="font-size: 0.85em; color: var(--danger, #e74c3c); margin-top: 4px;">${escapeHtml(s.error)}</div>` : ''}
                </div>`).join('');
        }

        function batchStatusLabel(s) {
            if (s.status === 'done') return '✅ 完成';
            if (s.status === 'processing') return '🎵 合成中…';
            if (s.status === 'failed') return '❌ 失败';
            return '⏳ 等待中';
        }

        async function startBatch() {
            const text = document.getElementById('batch-text').value.trim();
            if (!text) { alert('请先输入或上传长文本'); return; }
            const btn = document.getElementById('batch-generate-btn');
            const progress = document.getElementById('batch-progress');
            const status = document.getElementById('batch-status');
            btn.disabled = true;

            // 音色来源：优先上方已选音色，否则上传的参考音频
            const useVoice = document.getElementById('batch-save-voice').checked;
            let voiceId = selectedVoiceId || '';
            let promptWav = null;
            if (useVoice) {
                if (!voiceId && !promptFile) {
                    btn.disabled = false;
                    alert('请先在「合成模式」卡片中选择音色或上传参考音频');
                    return;
                }
                if (!voiceId && promptFile) promptWav = promptFile;
            } else {
                btn.disabled = false;
                alert('请勾选「使用上方所选音色」');
                return;
            }

            const formData = new FormData();
            formData.append('text', text);
            formData.append('speed', document.getElementById('batch-speed').value);
            formData.append('max_len', '100');
            if (voiceId) {
                formData.append('voice_id', voiceId);
                // 跟随页面所选模式；instruct 需指令文本
                if (currentMode === 'cross_lingual') formData.append('mode', 'cross_lingual');
                else if (currentMode === 'instruct') {
                    formData.append('mode', 'instruct');
                    formData.append('instruct_text', document.getElementById('instruct-text').value);
                } else {
                    formData.append('mode', 'zero_shot');
                    formData.append('prompt_text', document.getElementById('prompt-text').value || '');
                }
            } else if (promptWav) {
                formData.append('mode', 'zero_shot');
                formData.append('prompt_wav', promptWav);
                formData.append('prompt_text', document.getElementById('prompt-text').value || '');
            }

            progress.style.width = '5%';
            status.classList.remove('hidden');
            status.textContent = '⏳ 正在切分并创建任务…';
            document.getElementById('batch-download-all').classList.add('hidden');

            try {
                const res = await fetch('/api/tts/batch', { method: 'POST', body: formData });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || '创建任务失败');
                batchTaskId = data.task_id;
                batchDone = false;
                status.textContent = `🎬 任务已创建：共 ${data.total} 句，开始逐句合成…`;
                // 初始渲染所有句子为等待中
                renderBatchSentences(Array.from({ length: data.total }, (_, i) => ({ index: i, text: '', status: 'pending', filename: null, error: null })));
                pollBatch();
            } catch (e) {
                status.textContent = '❌ 创建任务失败: ' + e.message;
                progress.style.width = '0%';
                btn.disabled = false;
            }
        }

        async function pollBatch() {
            if (batchPollTimer) clearInterval(batchPollTimer);
            batchPollTimer = setInterval(async () => {
                if (!batchTaskId || batchDone) { clearInterval(batchPollTimer); return; }
                try {
                    const res = await fetch(`/api/tts/batch/${batchTaskId}`);
                    const data = await res.json();
                    if (!res.ok) throw new Error(data.detail || '查询失败');
                    const doneCount = data.sentences.filter(s => s.status === 'done').length;
                    const failCount = data.sentences.filter(s => s.status === 'failed').length;
                    document.getElementById('batch-progress').style.width = data.total ? `${Math.round(doneCount / data.total * 100)}%` : '0%';
                    const status = document.getElementById('batch-status');
                    status.classList.remove('hidden');
                    status.textContent = `⏳ ${doneCount}/${data.total} 句完成${failCount ? `（失败 ${failCount}）` : ''}…`;
                    renderBatchSentences(data.sentences);
                    if (data.status === 'completed' || data.status === 'failed' || doneCount + failCount >= data.total) {
                        clearInterval(batchPollTimer);
                        batchPollTimer = null;
                        batchDone = true;
                        const dlBtn = document.getElementById('batch-download-all');
                        if (doneCount > 0) {
                            dlBtn.classList.remove('hidden');
                            status.textContent = `${failCount ? '⚠️ 部分句子失败，' : '✅ '}${doneCount}/${data.total} 句已完成，可单独下载或一键打包下载`;
                        } else {
                            status.textContent = '❌ 全部句子合成失败';
                        }
                        document.getElementById('batch-generate-btn').disabled = false;
                    }
                } catch (e) {
                    clearInterval(batchPollTimer);
                    batchPollTimer = null;
                }
            }, 1500);
        }

        function downloadBatchZip() {
            if (!batchTaskId) return;
            window.location.href = `/api/tts/batch/${batchTaskId}/download`;
        }

        async function updateStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                const info = data.model_loaded 
                    ? `Model: ${data.model_dir} | GPU: ${data.gpu.memory_used}`
                    : 'Model not loaded';
                document.getElementById('gpu-info').textContent = info;
            } catch (e) {}
        }

        setLang('zh-CN');
        updateStatus();
        refreshVoices();
        setInterval(updateStatus, 30000);
    </script>
</body>
</html>'''

if __name__ == "__main__":
    # 启动守卫（2026-08-22 新增）：本服务只允许通过项目自带的 启动.bat 手动启动，
    # 防止被后台任务/其他程序自动拉起。_run_server.bat 会设置 YIJU_MANUAL_START=1。
    if os.getenv("YIJU_MANUAL_START") != "1":
        sys.stderr.write(
            "[启动守卫] 检测到非手动启动（缺少 YIJU_MANUAL_START=1 环境变量）。\n"
            "本服务只允许通过项目目录下的 启动.bat 手动启动，已拒绝启动。\n"
        )
        sys.exit(1)

    port = int(os.getenv("PORT", "8189"))

    # 重复启动守卫（2026-09-06 新增）：模型在 uvicorn lifespan 里加载且端口绑定在其后，
    # 第二个实例会先把 5GB+ 模型塞进显存、绑端口时才失败。先探测端口，被占用直接退出。
    import socket as _socket
    _probe = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    try:
        _probe.bind(("0.0.0.0", port))
    except OSError:
        print(
            f"[启动守卫] 端口 {port} 已被占用：服务已在运行，本窗口自动退出，不会重复加载模型。\n"
            f"直接用浏览器打开 http://localhost:{port} 即可使用。",
            flush=True,
        )
        sys.exit(0)
    finally:
        _probe.close()

    # 黑框实时日志（2026-09-06 新增）：_run_server.bat 不再把输出重定向进文件，
    # 而是由这里把 stdout/stderr 同步写往控制台（黑框可见运行状态）与
    # runtime\server.log（留档）。关黑框 = 进程被杀，日志同步停止。
    _log_file = os.getenv("YIJU_LOG_FILE")
    if _log_file:
        import re as _ansi_re
        _ansi_pattern = _ansi_re.compile(r"\x1b\[[0-9;]*m")

        class _Tee:
            """写穿到原流并追加到日志文件；文件侧剥离 ANSI 颜色码，写失败不影响控制台输出"""

            def __init__(self, stream, path):
                self._stream = stream
                self._file = open(path, "a", buffering=1, encoding="utf-8", errors="replace")

            def write(self, data):
                try:
                    self._file.write(_ansi_pattern.sub("", data))
                except Exception:
                    pass
                return self._stream.write(data)

            def flush(self):
                try:
                    self._file.flush()
                except Exception:
                    pass
                self._stream.flush()

            def isatty(self):
                # uvicorn 的日志着色器依赖 isatty() 判断是否加颜色码，必须透传
                try:
                    return self._stream.isatty()
                except Exception:
                    return False

            def writelines(self, lines):
                for line in lines:
                    self.write(line)

        Path(_log_file).parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = _Tee(sys.stdout, _log_file)
        sys.stderr = _Tee(sys.stderr, _log_file)
        print(f"[LOG] 控制台与日志文件双写已开启: {_log_file}", flush=True)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
