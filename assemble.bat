@echo off
rem ==================================================
rem  YIJUHUAKELONGSHENGYIN - one-key preflight for big assets
rem  Guarantee flow: (A) copy original project folder
rem  with big assets (fastest), or (B) clone this repo
rem  then run this script; details: see DEPLOY.md top.
rem ==================================================
setlocal
cd /d "%~dp0"
set "MISSING=0"
echo Checking required big assets...
if exist "code\pretrained_models\Fun-CosyVoice3-0.5B" (echo   OK   code\pretrained_models\Fun-CosyVoice3-0.5B) else (echo   MISS code\pretrained_models\Fun-CosyVoice3-0.5B ^& set MISSING=1)
if exist "code\cosyvoice" (echo   OK   code\cosyvoice) else (echo   MISS code\cosyvoice ^& set MISSING=1)
if exist "code\third_party\Matcha-TTS" (echo   OK   code\third_party\Matcha-TTS) else (echo   MISS code\third_party\Matcha-TTS ^& set MISSING=1)
echo.
if %MISSING%==0 (
  echo ALL big assets present. Run start.bat now.
) else (
  echo Some big assets missing. See DEPLOY.md (top section
  "Deployment guarantee") for download instructions.
)
pause
