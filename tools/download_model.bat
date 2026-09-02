@echo off
setlocal
rem ============================================================
rem  Download the Fun-CosyVoice3-0.5B model weights into
rem  code\pretrained_models\Fun-CosyVoice3-0.5B
rem  (about 7 GB, NOT shipped in this repo).
rem  Run from project root:  tools\download_model.bat
rem ============================================================
cd /d "%~dp0.."

if not exist "venv\Scripts\python.exe" (
    echo  [ERROR] venv not found. Run start.bat once first to install deps.
    pause
    exit /b 1
)

echo  Target : code\pretrained_models\Fun-CosyVoice3-0.5B
echo  Source : ModelScope  FunAudioLLM/Fun-CosyVoice3-0.5B
echo           https://modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B
echo  Size   : about 7 GB. Please keep the network on.
echo.
echo  Downloading ...
venv\Scripts\python.exe -c "from modelscope.hub.snapshot_download import snapshot_download; snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B', local_dir=r'code/pretrained_models/Fun-CosyVoice3-0.5B')"
if errorlevel 1 (
    echo  [ERROR] ModelScope download failed. Retry later, or use the
    echo          HuggingFace mirror manually:
    echo          huggingface-cli download FunAudioLLM/Fun-CosyVoice3-0.5B --local-dir code/pretrained_models/Fun-CosyVoice3-0.5B
    pause
    exit /b 1
)
if not exist "code\pretrained_models\Fun-CosyVoice3-0.5B\cosyvoice3.yaml" (
    echo  [ERROR] Snapshot incomplete (cosyvoice3.yaml missing).
    pause
    exit /b 1
)
echo.
echo  [OK] Model snapshot ready.
pause
exit /b 0
