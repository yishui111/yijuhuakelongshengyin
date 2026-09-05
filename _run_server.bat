@echo off
rem ============================================================
rem  CosyVoice service process (launched by start.bat, do not
rem  double-click): sets runtime env and runs code\app.py in venv
rem  Output shows live in this window AND is appended to
rem  runtime\server.log (tee is done inside app.py via YIJU_LOG_FILE).
rem  Closing this window stops the service and frees GPU memory.
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0code"

rem ---------- duplicate start guard: never load the model twice ----------
powershell -NoProfile -Command "$p = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'app\.py cosyvoice-app' }; if ($p) { exit 0 } else { exit 1 }" >nul 2>&1
if not errorlevel 1 (
    echo  [WARN] Service is ALREADY running. Refusing to start a second copy.
    echo         Just open http://localhost:8189 in your browser, or run status.bat
    pause
    exit /b 1
)

set INPUT_DIR=%~dp0input
set OUTPUT_DIR=%~dp0output
set VOICES_DIR=%~dp0voices
set MODEL_DIR=pretrained_models\Fun-CosyVoice3-0.5B
set MODELSCOPE_CACHE=%~dp0cache\modelscope
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set PYTHONUNBUFFERED=1
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
set PORT=8189
set YIJU_MANUAL_START=1
set YIJU_LOG_FILE=%~dp0runtime\server.log

if not exist "%~dp0runtime" mkdir "%~dp0runtime"
if exist "%~dp0runtime\server.log" copy /y "%~dp0runtime\server.log" "%~dp0runtime\server.log.old" >nul 2>&1
..\venv\Scripts\python.exe -X utf8 app.py cosyvoice-app

echo.
echo  [EXIT] Service process has ended. It is safe to close this window.
pause
