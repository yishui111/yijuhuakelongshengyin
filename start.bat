@echo off
setlocal
rem ============================================================
rem  CosyVoice one-sentence voice clone - Windows launcher
rem  First run: creates venv, installs dependencies, fetches the
rem  third-party engine code and the model weights when missing.
rem  Uses %~dp0 for its own location (any path is fine).
rem ============================================================
cd /d "%~dp0"
title CosyVoice Voice Clone

rem ---------- 1. locate Python 3.10 - 3.13 ----------
rem (Python 3.14 has no Windows wheels for kaldifst/wetext)
set "PY="
py -V >nul 2>&1
if %errorlevel%==0 (
    for %%V in (3.13 3.12 3.11 3.10) do (
        py -%%V -c "import sys" >nul 2>&1
        if not errorlevel 1 (
            set "PY=py -%%V"
            goto pyfound
        )
    )
)
python -V >nul 2>&1
if %errorlevel%==0 (
    python -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PY=python"
        goto pyfound
    )
)
goto pynotfound

:pyfound
%PY% -V
goto pyok

:pynotfound
echo.
echo  [ERROR] Python 3.10 - 3.13 not found.
echo          Install Python and tick the "py launcher" option:
echo          https://www.python.org/downloads/
echo.
pause
exit /b 1

:pyok

rem ---------- 2. first run: create virtual env ----------
if exist "venv\Scripts\python.exe" goto venvok
echo.
echo  [STEP] Creating Python virtual environment (venv) ...
%PY% -m venv venv
if errorlevel 1 (
    echo  [ERROR] Failed to create venv.
    pause
    exit /b 1
)
:venvok

rem ---------- 3. first run: install dependencies ----------
if exist "venv\.deps_installed" goto depsok
echo.
echo  [STEP] Installing dependencies (torch ~3 GB download,
echo         takes 5-15 minutes, keep network on) ...
venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 (
    echo  [ERROR] pip upgrade failed.
    pause
    exit /b 1
)
rem setuptools < 81 keeps pkg_resources (openai-whisper build needs it)
venv\Scripts\python.exe -m pip install "setuptools==80.9.0" wheel
if errorlevel 1 (
    echo  [ERROR] setuptools install failed.
    pause
    exit /b 1
)
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo  [ERROR] requirements install failed, check messages above.
    pause
    exit /b 1
)
rem openai-whisper needs --no-build-isolation (uses pkg_resources)
venv\Scripts\python.exe -m pip install openai-whisper==20231117 --no-build-isolation
if errorlevel 1 (
    echo  [ERROR] openai-whisper install failed.
    pause
    exit /b 1
)
echo ok > venv\.deps_installed
echo  [OK] Dependencies installed.
:depsok

rem ---------- 4. third-party engine code (not in repo) ----------
if not exist "code\cosyvoice\cli\cosyvoice.py" goto needengine
if not exist "code\third_party\Matcha-TTS\matcha\__init__.py" goto needengine
goto haveengine
:needengine
echo.
echo  [STEP] Third-party engine code is missing. Fetching now
echo         (CosyVoice engine + Matcha-TTS) ...
call "%~dp0tools\download_engine.bat"
if errorlevel 1 (
    echo  [ERROR] Engine fetch failed, see messages above.
    pause
    exit /b 1
)
:haveengine

rem ---------- 5. model weights (not in repo) ----------
if exist "code\pretrained_models\Fun-CosyVoice3-0.5B\cosyvoice3.yaml" goto havemodel
echo.
echo  [WARN] Model weights not found:
echo         code\pretrained_models\Fun-CosyVoice3-0.5B
echo         They are NOT shipped in this repo (about 7 GB).
echo         Source: https://modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B
echo         (HuggingFace mirror: FunAudioLLM/Fun-CosyVoice3-0.5B)
set /p DL=Download now via tools\download_model.bat ? [y/N]: 
if /i "%DL%"=="y" (
    call "%~dp0tools\download_model.bat"
    if errorlevel 1 (
        pause
        exit /b 1
    )
    if not exist "code\pretrained_models\Fun-CosyVoice3-0.5B\cosyvoice3.yaml" (
        echo  [ERROR] Model still missing after download.
        pause
        exit /b 1
    )
) else (
    echo.
    echo  Please run:  tools\download_model.bat
    pause
    exit /b 1
)
:havemodel

rem ---------- 6. already running? ----------
powershell -NoProfile -Command "$p = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'app\.py cosyvoice-app' }; if ($p) { exit 0 } else { exit 1 }" >nul 2>&1
if not errorlevel 1 (
    echo  [INFO] Service is already running: http://localhost:8188
    pause
    exit /b 0
)

rem ---------- 7. start service in a new window ----------
if not exist "runtime" mkdir runtime
echo  [START] Launching service (log: runtime\server.log) ...
start "CosyVoice-Service" /min "%~dp0_run_server.bat"

rem ---------- 8. wait for model readiness ----------
echo  [WAIT] Model loading, about 1-3 minutes, please wait ...
set /a n=0
:wait
set /a n+=1
if %n% gtr 130 (
    echo  [ERROR] Timeout waiting for the model. Check runtime\server.log.
    pause
    exit /b 1
)
powershell -NoProfile -Command "try { $r = Invoke-RestMethod -Uri 'http://localhost:8188/health' -TimeoutSec 3; if ($r.status -eq 'healthy') { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 goto ready
timeout /t 5 /nobreak >nul
goto wait

:ready
echo.
echo  ================================================
echo   Startup OK!
echo   Web UI / API : http://localhost:8188
echo   API docs     : http://localhost:8188/docs
echo   Log          : runtime\server.log
echo  ================================================
echo.
start "" "http://localhost:8188"
pause
exit /b 0
