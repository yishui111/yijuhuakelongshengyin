# CosyVoice 功能测试脚本
# 用法: powershell -ExecutionPolicy Bypass -File .\tests\test_tts.ps1
param([string]$BaseUrl = "http://localhost:8188")
$ErrorActionPreference = "Stop"
$outDir = Join-Path $PSScriptRoot "output"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

Write-Host "[1/4] 健康检查..."
$health = Invoke-RestMethod -Uri "$BaseUrl/health" -TimeoutSec 30
if ($health.status -ne "healthy") { throw "服务未就绪: $($health | ConvertTo-Json -Compress)" }
Write-Host "   OK model_loaded=$($health.gpu.model_loaded) GPU=$($health.gpu.gpu.memory_used)/$($health.gpu.gpu.memory_total)"

Write-Host "[2/4] 音色列表..."
$voices = Invoke-RestMethod -Uri "$BaseUrl/v1/voices" -TimeoutSec 30
$voiceId = $null
if ($voices.custom_voices -and $voices.custom_voices.Count -gt 0) { $voiceId = $voices.custom_voices[0].id }
if (-not $voiceId -and $voices.preset_voices -and $voices.preset_voices.Count -gt 0) { $voiceId = $voices.preset_voices[0] }
if (-not $voiceId) { throw "没有可用音色" }
Write-Host "   使用音色: $voiceId"

Write-Host "[3/4] 合成语音 (voice=$voiceId)..."
$body = @{ model = "cosyvoice-v3"; input = "你好，这是一次声音克隆功能测试。"; voice = $voiceId; response_format = "wav"; speed = 1.0 } | ConvertTo-Json
$out = Join-Path $outDir ("test_tts_{0}.wav" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
Invoke-RestMethod -Uri "$BaseUrl/v1/audio/speech" -Method Post -ContentType "application/json; charset=utf-8" -Body $body -OutFile $out -TimeoutSec 300
$size = (Get-Item $out).Length
if ($size -lt 1000) { throw "输出文件异常小: $size bytes" }
Write-Host "   输出: $out ($size bytes)"

Write-Host "[4/4] 结果: PASS"
exit 0
