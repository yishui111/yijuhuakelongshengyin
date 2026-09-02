"""
CosyVoice MCP Server - Model Context Protocol Interface
"""
import os
import sys
import gc
import time
import uuid
import threading
from pathlib import Path
from typing import Optional

import torch
import torchaudio

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / "third_party/Matcha-TTS"))

from fastmcp import FastMCP
from cosyvoice.cli.cosyvoice import AutoModel

# Directories
INPUT_DIR = Path(os.getenv("INPUT_DIR", "/data/input"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/data/output"))
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# GPU Manager (shared singleton)
class GPUManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance
    
    def _init(self):
        self.model = None
        self.model_dir = None
        self.lock = threading.Lock()
        self.last_used = time.time()
        self.idle_timeout = int(os.getenv("GPU_IDLE_TIMEOUT", "600"))
        
    def get_model(self, model_dir: str = None):
        with self.lock:
            if model_dir is None:
                model_dir = os.getenv("MODEL_DIR", "pretrained_models/CosyVoice2-0.5B")
            if self.model is None or self.model_dir != model_dir:
                self._load_model(model_dir)
            self.last_used = time.time()
            return self.model
    
    def _load_model(self, model_dir: str):
        if self.model is not None:
            self.offload()
        print(f"[MCP] Loading model from {model_dir}...")
        self.model = AutoModel(model_dir=model_dir)
        self.model_dir = model_dir
    
    def offload(self):
        if self.model:
            del self.model
            self.model = None
            self.model_dir = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[MCP] GPU memory released")
    
    def status(self) -> dict:
        gpu_info = {"available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            gpu_info.update({
                "device": torch.cuda.get_device_name(0),
                "memory_used_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
                "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            })
        return {
            "model_loaded": self.model is not None,
            "model_dir": self.model_dir,
            "gpu": gpu_info,
            "idle_seconds": int(time.time() - self.last_used) if self.model else None
        }

gpu_manager = GPUManager()

# MCP Server
mcp = FastMCP("CosyVoice")

@mcp.tool()
def tts_zero_shot(
    text: str,
    prompt_text: str,
    prompt_audio_path: str,
    speed: float = 1.0,
    output_filename: Optional[str] = None
) -> dict:
    """
    Zero-shot voice cloning TTS.
    
    Args:
        text: Text to synthesize
        prompt_text: Text content of the prompt audio (must match audio)
        prompt_audio_path: Path to reference audio file (3-30 seconds)
        speed: Speech speed (0.5-2.0, default 1.0)
        output_filename: Optional output filename (auto-generated if not provided)
    
    Returns:
        dict with status, output_path, duration_seconds
    """
    try:
        model = gpu_manager.get_model()
        
        speeches = []
        for chunk in model.inference_zero_shot(text, prompt_text, prompt_audio_path, stream=False, speed=speed):
            speeches.append(chunk['tts_speech'])
        
        full_speech = torch.cat(speeches, dim=1)
        filename = output_filename or f"tts_{uuid.uuid4().hex}.wav"
        output_path = OUTPUT_DIR / filename
        torchaudio.save(str(output_path), full_speech, model.sample_rate)
        
        duration = full_speech.shape[1] / model.sample_rate
        
        return {
            "status": "success",
            "output_path": str(output_path),
            "duration_seconds": round(duration, 2),
            "sample_rate": model.sample_rate
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def tts_cross_lingual(
    text: str,
    prompt_audio_path: str,
    speed: float = 1.0,
    output_filename: Optional[str] = None
) -> dict:
    """
    Cross-lingual voice cloning TTS.
    
    Args:
        text: Text to synthesize (can be different language from prompt)
        prompt_audio_path: Path to reference audio file
        speed: Speech speed (0.5-2.0)
        output_filename: Optional output filename
    
    Returns:
        dict with status, output_path, duration_seconds
    """
    try:
        model = gpu_manager.get_model()
        
        speeches = []
        for chunk in model.inference_cross_lingual(text, prompt_audio_path, stream=False, speed=speed):
            speeches.append(chunk['tts_speech'])
        
        full_speech = torch.cat(speeches, dim=1)
        filename = output_filename or f"tts_{uuid.uuid4().hex}.wav"
        output_path = OUTPUT_DIR / filename
        torchaudio.save(str(output_path), full_speech, model.sample_rate)
        
        return {
            "status": "success",
            "output_path": str(output_path),
            "duration_seconds": round(full_speech.shape[1] / model.sample_rate, 2)
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def tts_instruct(
    text: str,
    instruct_text: str,
    prompt_audio_path: str,
    speed: float = 1.0,
    output_filename: Optional[str] = None
) -> dict:
    """
    Instruction-controlled TTS with voice cloning.
    
    Args:
        text: Text to synthesize
        instruct_text: Instruction for style control (e.g., "用四川话说", "speak slowly")
        prompt_audio_path: Path to reference audio file
        speed: Speech speed (0.5-2.0)
        output_filename: Optional output filename
    
    Returns:
        dict with status, output_path, duration_seconds
    """
    try:
        model = gpu_manager.get_model()
        
        speeches = []
        if hasattr(model, 'inference_instruct2'):
            output = model.inference_instruct2(text, instruct_text, prompt_audio_path, stream=False, speed=speed)
        else:
            output = model.inference_instruct(text, "", instruct_text, stream=False, speed=speed)
        
        for chunk in output:
            speeches.append(chunk['tts_speech'])
        
        full_speech = torch.cat(speeches, dim=1)
        filename = output_filename or f"tts_{uuid.uuid4().hex}.wav"
        output_path = OUTPUT_DIR / filename
        torchaudio.save(str(output_path), full_speech, model.sample_rate)
        
        return {
            "status": "success",
            "output_path": str(output_path),
            "duration_seconds": round(full_speech.shape[1] / model.sample_rate, 2)
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def tts_sft(
    text: str,
    speaker_id: str,
    speed: float = 1.0,
    output_filename: Optional[str] = None
) -> dict:
    """
    TTS using pretrained speaker voice.
    
    Args:
        text: Text to synthesize
        speaker_id: Speaker ID (use list_speakers to get available IDs)
        speed: Speech speed (0.5-2.0)
        output_filename: Optional output filename
    
    Returns:
        dict with status, output_path, duration_seconds
    """
    try:
        model = gpu_manager.get_model()
        
        speeches = []
        for chunk in model.inference_sft(text, speaker_id, stream=False, speed=speed):
            speeches.append(chunk['tts_speech'])
        
        full_speech = torch.cat(speeches, dim=1)
        filename = output_filename or f"tts_{uuid.uuid4().hex}.wav"
        output_path = OUTPUT_DIR / filename
        torchaudio.save(str(output_path), full_speech, model.sample_rate)
        
        return {
            "status": "success",
            "output_path": str(output_path),
            "duration_seconds": round(full_speech.shape[1] / model.sample_rate, 2)
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def list_speakers() -> dict:
    """
    List available pretrained speaker voices.
    
    Returns:
        dict with speakers list
    """
    try:
        model = gpu_manager.get_model()
        return {"status": "success", "speakers": model.list_available_spks()}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def gpu_status() -> dict:
    """
    Get current GPU and model status.
    
    Returns:
        dict with model_loaded, model_dir, gpu info, idle_seconds
    """
    return gpu_manager.status()

@mcp.tool()
def gpu_offload() -> dict:
    """
    Release GPU memory by unloading the model.
    
    Returns:
        dict with status message
    """
    gpu_manager.offload()
    return {"status": "success", "message": "GPU memory released"}

@mcp.tool()
def load_model(model_dir: str) -> dict:
    """
    Load a specific model.
    
    Args:
        model_dir: Path to model directory (e.g., "pretrained_models/CosyVoice2-0.5B")
    
    Returns:
        dict with status and loaded model info
    """
    try:
        gpu_manager.get_model(model_dir)
        return {"status": "success", "model_dir": model_dir}
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    mcp.run()
