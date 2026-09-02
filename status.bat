@echo off
setlocal
rem ============================================================
rem  CosyVoice voice clone - show service / GPU status
rem ============================================================
cd /d "%~dp0"
echo.
echo  CosyVoice Voice Clone - Status
echo.
powershell -NoProfile -Command "& { try { $r = Invoke-RestMethod -Uri 'http://localhost:8188/health' -TimeoutSec 5; Write-Host ('  Status       : ' + $r.status); Write-Host ('  Model loaded : ' + $r.gpu.model_loaded); Write-Host ('  Model dir    : ' + $r.gpu.model_dir); if ($r.gpu.gpu.available) { Write-Host ('  GPU          : ' + $r.gpu.gpu.device + ' | ' + $r.gpu.gpu.memory_used + ' / ' + $r.gpu.gpu.memory_total) } else { Write-Host '  GPU          : not available (CPU only?)' } } catch { Write-Host '  Service      : not reachable (stopped or still starting)' }; $p = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'app\.py cosyvoice-app' }; if ($p) { Write-Host ('  Process      : PID ' + ($p.ProcessId -join ', ') + ' running') } else { Write-Host '  Process      : none' } }"
echo.
pause
exit /b 0
