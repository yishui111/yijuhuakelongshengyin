@echo off
setlocal
rem ============================================================
rem  CosyVoice voice clone - stop the running service
rem  Kills the python.exe started by start.bat / _run_server.bat
rem  (identified by the command line marker "app.py cosyvoice-app")
rem ============================================================
cd /d "%~dp0"
echo.
echo  CosyVoice Voice Clone - Stop
echo.
powershell -NoProfile -Command "& { $p = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'app\.py cosyvoice-app' }; if ($p) { $p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; Write-Host ('  Stopped service (PID ' + $_.ProcessId + ')') } } else { Write-Host '  Service is not running.' } }"
echo.
pause
exit /b 0
