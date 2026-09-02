# -*- coding: utf-8 -*-
"""TTS 原生环境排障/回归测试脚本（绕开 Web 服务，直接加载模型合成一句）。

前置条件（见 DEPLOY.md）：
  1. code\pretrained_models\Fun-CosyVoice3-0.5B 模型快照已就位；
  2. 引擎代码已就位：code\cosyvoice、code\third_party\Matcha-TTS；
  3. venv 依赖已安装。

用法（在项目根目录）：
    venv\Scripts\python.exe -X utf8 tests\tts_native_test.py

参考音色自动取 voices\voices.json 中的第一个自定义音色；
若还没有任何音色，脚本会提示先到 Web 页面创建（或手动放置
voices\<id>\prompt.wav 并在 voices.json 登记）。输出 wav 写到 tests\output\。
"""
import json
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = ROOT / "code"
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "third_party" / "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402


def find_first_voice():
    """从 voices/voices.json 里找第一个音频文件仍存在的音色"""
    index = ROOT / "voices" / "voices.json"
    if not index.exists():
        return None
    data = json.loads(index.read_text(encoding="utf-8"))
    for vid, v in data.items():
        p = Path(v.get("audio_path") or (ROOT / "voices" / vid / "prompt.wav"))
        if p.exists():
            return vid, v.get("text", ""), p
    return None


def main():
    print("=== loading model ===", flush=True)
    model = AutoModel(model_dir=str(CODE_DIR / "pretrained_models" / "Fun-CosyVoice3-0.5B"))
    print("=== TTS test ===", flush=True)

    voice = find_first_voice()
    if voice is None:
        print(
            "[SKIP] no saved voice found: create a clone voice on the Web page "
            "first, or place voices/<id>/prompt.wav and register it in voices/voices.json",
            flush=True,
        )
        return
    vid, voice_text, voice_wav = voice
    print(f"    voice: {vid} ({voice_wav.name})", flush=True)

    try:
        output = model.inference_zero_shot(
            "你好，这是一段测试语音，验证原生环境能否正常合成。",
            voice_text,
            str(voice_wav),
            stream=False,
            speed=1.0,
        )
        speeches = [chunk["tts_speech"] for chunk in output]
        full = torch.cat(speeches, dim=1)
        import soundfile as sf

        out_dir = ROOT / "tests" / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"test_native_{time.strftime('%Y%m%d_%H%M%S')}.wav"
        sf.write(str(out), full.squeeze(0).detach().cpu().numpy(), model.sample_rate, subtype="PCM_16")
        print(f"=== TTS done, wav shape: {tuple(full.shape)} -> {out} ===", flush=True)
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()
