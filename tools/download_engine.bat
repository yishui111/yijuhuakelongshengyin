@echo off
setlocal
rem ============================================================
rem  Fetch third-party engine code needed at runtime.
rem  NOT shipped in this repo (third-party open source):
rem    1) cosyvoice package  <- https://github.com/FunAudioLLM/CosyVoice
rem    2) Matcha-TTS         <- https://github.com/shivammehta25/Matcha-TTS
rem  Run from project root:  tools\download_engine.bat
rem ============================================================
cd /d "%~dp0.."
set "DEST=%CD%\code"
set "TMPD=%TEMP%\cosyvoice_engine_src"

git --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] git not found. Install Git for Windows:
    echo          https://git-scm.com/download/win
    echo          then run tools\download_engine.bat again.
    exit /b 1
)

if exist "%DEST%\cosyvoice\cli\cosyvoice.py" goto havecosy
echo  [1/2] Downloading CosyVoice engine code (cosyvoice package) ...
if exist "%TMPD%" rmdir /s /q "%TMPD%"
git clone --depth 1 --filter=blob:none --sparse https://github.com/FunAudioLLM/CosyVoice.git "%TMPD%"
if errorlevel 1 (
    echo  [ERROR] CosyVoice clone failed. Check network and retry.
    exit /b 1
)
cd /d "%TMPD%"
git sparse-checkout set cosyvoice
if errorlevel 1 (
    echo  [WARN] Sparse checkout failed, falling back to full shallow clone ...
    cd /d "%TEMP%"
    if exist "%TMPD%" rmdir /s /q "%TMPD%"
    git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git "%TMPD%"
    if errorlevel 1 (
        echo  [ERROR] CosyVoice clone failed again. Check network and retry.
        exit /b 1
    )
)
if not exist "%TMPD%\cosyvoice\cli\cosyvoice.py" (
    echo  [ERROR] Unexpected clone layout (cosyvoice package not found).
    exit /b 1
)
if exist "%DEST%\cosyvoice" rmdir /s /q "%DEST%\cosyvoice"
xcopy /e /i /y "%TMPD%\cosyvoice" "%DEST%\cosyvoice" >nul
if errorlevel 1 (
    echo  [ERROR] Failed to copy cosyvoice package.
    exit /b 1
)
echo  [OK] cosyvoice package installed to code\cosyvoice
:havecosy

if exist "%DEST%\third_party\Matcha-TTS\matcha\__init__.py" goto havematcha
echo  [2/2] Downloading Matcha-TTS ...
if not exist "%DEST%\third_party" mkdir "%DEST%\third_party"
git clone --depth 1 https://github.com/shivammehta25/Matcha-TTS.git "%DEST%\third_party\Matcha-TTS"
if errorlevel 1 (
    echo  [ERROR] Matcha-TTS clone failed. Check network and retry.
    exit /b 1
)
echo  [OK] Matcha-TTS installed to code\third_party\Matcha-TTS
:havematcha

if exist "%TMPD%" rmdir /s /q "%TMPD%"
echo.
echo  Done. Engine code is ready. Rerun start.bat to continue.
exit /b 0
