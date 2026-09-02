# tests — 功能测试

| 脚本 | 作用 | 需要服务已启动 |
| ---- | ---- | ---- |
| `test_tts.ps1` | 健康检查 + 音色列表 + OpenAI 兼容 TTS 合成，wav 输出到 `tests\output\` | ✅ |
| `batch_tts_test.py` | **长文本分段合成**：切分预览 / 创建逐句任务 / 轮询进度 / 一键下载 zip | ✅ |
| `tts_native_test.py` | 绕开 Web 服务直接加载模型合成一句（排障/回归用） | ❌（需引擎+模型+venv） |

## 用法

先启动服务（双击 `启动.bat` 或 `start.bat`，等模型加载完成），再执行：

```powershell
# 健康 + 音色 + 单句合成
powershell -ExecutionPolicy Bypass -File .\tests\test_tts.ps1

# 长文本分段合成（自动取第一个自定义音色；没有时先到页面创建一个）
venv\Scripts\python.exe -X utf8 tests\batch_tts_test.py
```

`tts_native_test.py` 适合服务起不来时直接验证模型/引擎环境：

```powershell
venv\Scripts\python.exe -X utf8 tests\tts_native_test.py
```

> 输出统一写到 `tests\output\`（已在 .gitignore 中忽略，可随时删除）。
