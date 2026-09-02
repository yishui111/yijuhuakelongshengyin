@echo off
rem ============================================================
rem  CosyVoice service process (launched by start.bat, do not
rem  double-click): sets runtime env and runs code\app.py in venv
rem ============================================================
cd /d "%~dp0code"

set INPUT_DIR=%~dp0input
set OUTPUT_DIR=%~dp0output
set VOICES_DIR=%~dp0voices
set MODEL_DIR=pretrained_models\Fun-CosyVoice3-0.5B
set MODELSCOPE_CACHE=%~dp0cache\modelscope
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set PYTHONUNBUFFERED=1
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
set PORT=8188
set YIJU_MANUAL_START=1

if not exist "%~dp0runtime" mkdir "%~dp0runtime"
if exist "%~dp0runtime\server.log" copy /y "%~dp0runtime\server.log" "%~dp0runtime\server.log.old" >nul 2>&1
..\venv\Scripts\python.exe -X utf8 app.py cosyvoice-app >> "%~dp0runtime\server.log" 2>&1
