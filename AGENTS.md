# AGENTS.md — 一句话克隆声音（CosyVoice 声音克隆 / TTS 服务）项目档案与维护约定

> ⚠️ **修改本仓库前请先通读本文件**：本文件是给「AI 助手 / 开发者」看的项目记忆，记录项目情况、本仓库内容边界与维护约定，避免重复勘探、误删误传。用户向文档见 README.md（使用）、DEPLOY.md（部署）、使用说明.md（页面操作）。

## 1. 项目定位（一句话）

录一句 3-30 秒参考音频 + 参考文本 → 克隆出该音色 → 输入任意文本合成语音的**本地 Web 服务**（FastAPI + 内嵌中文 Web UI，零样本克隆基于阿里 Fun-CosyVoice3-0.5B，Windows + NVIDIA GPU 原生运行，无需 Docker）。

## 2. 架构与组件

| 组件 | 作用 |
| ---- | ---- |
| `code/app.py` | **主服务**（自研，单文件 ~2000 行）：FastAPI（端口 8188）+ 内嵌 Web 控制台 HTML + 零样本克隆/长文本分段合成/ASR 质量自检/OpenAI 兼容 API；带启动守卫（需 env `YIJU_MANUAL_START=1`，由 `_run_server.bat` 设置，防后台误拉起） |
| `code/model.py` | Fun-ASR-Nano-2512 的远程代码（`funasr.AutoModel(..., remote_code="./model.py")` 按 CWD 加载，**必需**；模型快照里没有该文件） |
| `code/mcp_server.py` | 可选 MCP 服务（实验性，需 `pip install fastmcp`；默认模型目录 CosyVoice2-0.5B 可经 env `MODEL_DIR` 覆盖） |
| `code/cosyvoice/` `code/third_party/` | ⚠️ **第三方引擎，不入库**：部署时由 `tools\download_engine.bat` 拉取（FunAudioLLM/CosyVoice 的 cosyvoice 包 + shivammehta25/Matcha-TTS） |
| `code/pretrained_models/` | ⚠️ **模型权重，不入库**：`tools\download_model.bat` 下载 Fun-CosyVoice3-0.5B 到 `code\pretrained_models\Fun-CosyVoice3-0.5B` |
| `tools/download_engine.bat` `download_model.bat` | 首次部署下载器（ASCII bat，start.bat 自动调用/提示） |
| `tools/extract_image.py` | 历史工具（从旧 image.tar 提取文件），image.tar 已不随仓库分发，仅存档 |
| `start.bat` `stop.bat` `status.bat` | 一键启停/状态（ASCII 实现）；`启动.bat` `关闭.bat` `状态.bat` 为同名包装（内容纯 ASCII，仅文件名中文） |
| `_run_server.bat` | 服务进程启动器：设 env（INPUT/OUTPUT/VOICES_DIR、MODEL_DIR、MODELSCOPE_CACHE、PORT=8188、YIJU_MANUAL_START=1）→ `venv\python -X utf8 app.py cosyvoice-app` → 日志 `runtime\server.log` |
| `tests/` | 回归测试（需服务运行：test_tts.ps1 / batch_tts_test.py；tts_native_test.py 直连引擎排障） |
| `requirements.txt` | 依赖锁定（torch 2.11.0+cu128、funasr 1.2.9、fastapi 等）；`openai-whisper==20231117` 由 start.bat 单独 `--no-build-isolation` 安装 |
| `docker-compose.yml` | ⚠️ 旧 Docker 方案遗留（存档，已不使用） |

- 技术栈：Python 3.10-3.13 + FastAPI + PyTorch(CUDA) + Fun-CosyVoice3-0.5B + Fun-ASR-Nano；无前端构建，UI 内嵌 app.py
- 端口：**8188**（Web/API：http://localhost:8188 ，Swagger：/docs）
- 数据目录：`voices\` 音色库、`output\` 合成 wav、`input\` 临时、`cache\modelscope` 模型缓存、`runtime\` 日志
- 启动链：`start.bat` →（建 venv/装依赖/拉引擎/提示下模型）→ `_run_server.bat`（新窗口）→ app.py 载入模型约 1-3 分钟 → `/health` healthy → 开浏览器
- 停止：`stop.bat` 按进程命令行含 `app.py cosyvoice-app` 精确匹配停止（防误杀其他 python）

## 3. 本仓库 = GitHub 公开裁剪版（重要边界）

以下内容刻意不入库（.gitignore 已覆盖）：

- 模型/引擎：`code/pretrained_models/`、`code/cosyvoice/`、`code/third_party/`（部署时下载）
- 素材/隐私：`voices/`（音色库含真人参考音频与索引）、`ziliao/`、`static/`（引擎示例音频副本）、`code/asset/`
- 产物/缓存/运行时：`input/ output/ cache/ runtime/ demucs_models/ tts_test_tmp/ venv/ image.tar`、日志、`tests/output/`
- 本仓库保留的代码中**不含密钥**；所有 `.bat` 均为纯 ASCII + CRLF + 无 BOM

> 规则：新增内容不得引入密钥/真人素材/模型权重/大件；仓库应能按 DEPLOY.md 从零复现。

## 4. 跨项目依赖 / 机器特定配置

- 运行时依赖（联网下载，见 DEPLOY.md）：Fun-CosyVoice3-0.5B 权重（ModelScope/HF，~7GB）、Fun-ASR-Nano-2512（首启自动，~2GB）、CosyVoice 引擎源码与 Matcha-TTS（github clone）、torch cu128（pip，~3GB）
- 机器约束：NVIDIA 驱动 ≥ 570；Python 3.10-3.13（3.14 无 kaldifst/wetext 轮子）
- app.py 对引擎做了一处兼容：读取引擎 frontend 的 `prompt_cache` 一律走 `_engine_prompt_cache_size()`（getattr），官方原版引擎无该属性也不会报错（作者本机引擎打过补丁）

## 5. 已知问题 / TODO / 安全注意

- 引擎源码由部署脚本从 GitHub 拉取「当前 master」，若上游接口变动可能需同步适配 app.py（本机验过的引擎版本为镜像内快照 + 少量本地补丁）
- `mcp_server.py` 默认指向 CosyVoice2-0.5B（与主服务 v3 不同），用前需自行准备对应权重目录
- 声音克隆需遵守本人/授权声音使用规范
- README 徽章与 clone 地址中的 `yishui111` 在推送前统一替换

## 6. 维护更新约定

- 改动代码后：同步更新 README.md 与 DEPLOY.md；bat 改动需保持纯 ASCII + CRLF + 无 BOM（用 `%~dp0` 定位，不硬编码盘符）
- 提交：`git add . && git commit -m "..." && git push origin main`（本裁剪版仓库由 .dsh_upload_prep 流程生成，勿把原项目目录直接推上去）
- 中文文档用 UTF-8（无 BOM）；模型/引擎版本或端口变化时更新本文件与 DEPLOY.md
