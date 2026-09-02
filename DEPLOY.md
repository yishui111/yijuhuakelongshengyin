
## 🚀 换电脑部署（保证可用）

> **方式 A（推荐 · 100% 保证）**：用 U 盘 / 网盘把「原项目整份文件夹」（含全部大件）复制到新电脑 → 双击 `start.bat` 即可。
>
> **方式 B（代码装配）**：`git clone` 本仓库 → 双击 `assemble.bat` 预检大件 → 按提示补齐缺失项（下载地址见下文/README）→ 双击 `start.bat`。

> 说明：引擎、模型、镜像、运行时等大件体积超过 GitHub 单文件 100MB 上限，**不随仓库分发**；本仓库承载全部自研代码与装配指引，"方式 A"是换机部署最稳路径，"方式 B"适合需要重新下载大件的场景。
# 部署方案（DEPLOY.md）— 一句话克隆声音（CosyVoice 声音克隆 / TTS 服务）

> 目标：把本项目复制到另一台 Windows 电脑，按本文操作即可启动。
> 本仓库只含自研代码/脚本/文档；**模型权重、第三方引擎源码、运行缓存均需按本文下载生成**。
> 项目原名项目：CosyVoice 声音克隆 — Windows 原生部署（无需 Docker）。

## 一、环境要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10/11 x64（中文或英文均可） |
| GPU | NVIDIA 显卡，显存 ≥ 8GB（推荐 16GB） |
| NVIDIA 驱动 | ≥ 570（支持 CUDA 12.8，torch 2.11+cu128） |
| Python | 3.10 - 3.13（**不能用 3.14**：kaldifst/wetext 无 3.14 Windows 轮子），勾选 "py launcher" |
| Git | Git for Windows（拉取第三方引擎源码用） |
| 磁盘 / 网络 | ≥ 25GB 空闲；可访问 github.com 与 modelscope.cn（可配置镜像） |

## 二、复制项目

把整个项目文件夹（本仓库内容）复制/克隆到目标机器任意目录（路径可含空格，脚本用 `%~dp0` 自适应）。

> 说明：仓库内 `code\cosyvoice`、`code\third_party`、`code\pretrained_models`、`venv`、`voices`、`cache` 等目录**不在仓库中**，由以下步骤生成。

## 三、首次部署（推荐：一条命令自动完成）

在项目根目录双击 **`start.bat`**（中文系统也可双击 **`启动.bat`**），脚本按顺序自动完成：

1. **定位 Python 3.10-3.13**：依次尝试 `py -3.13/-3.12/-3.11/-3.10`、`python`（版本不符会提示安装）；
2. **创建虚拟环境**：`python -m venv venv`；
3. **安装依赖**（torch ~3GB，5-15 分钟，需保持联网）：
   ```bat
   venv\Scripts\python.exe -m pip install --upgrade pip
   venv\Scripts\python.exe -m pip install "setuptools==80.9.0" wheel
   venv\Scripts\python.exe -m pip install -r requirements.txt
   venv\Scripts\python.exe -m pip install openai-whisper==20231117 --no-build-isolation
   ```
   （`openai-whisper` 必须 `--no-build-isolation`：其构建脚本依赖 pkg_resources，新版 setuptools 已移除；安装成功会写 `venv\.deps_installed` 标记，之后不再重复安装）
4. **拉取第三方引擎代码**（自动调用 `tools\download_engine.bat`）：
   ```bat
   :: cosyvoice 引擎包（Apache-2.0）→ code\cosyvoice
   git clone --depth 1 --filter=blob:none --sparse https://github.com/FunAudioLLM/CosyVoice.git %TEMP%\cosyvoice_engine_src
   cd %TEMP%\cosyvoice_engine_src && git sparse-checkout set cosyvoice
   xcopy /e /i /y %TEMP%\cosyvoice_engine_src\cosyvoice code\cosyvoice

   :: Matcha-TTS（MIT）→ code\third_party\Matcha-TTS
   git clone --depth 1 https://github.com/shivammehta25/Matcha-TTS.git code\third_party\Matcha-TTS
   ```
5. **下载模型权重**（按提示运行 `tools\download_model.bat`，或手动执行下面任一条）：
   ```bash
   :: 方式一：ModelScope（推荐，国内快）
   venv\Scripts\python.exe -c "from modelscope.hub.snapshot_download import snapshot_download; snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B', local_dir=r'code/pretrained_models/Fun-CosyVoice3-0.5B')"

   :: 方式二：HuggingFace 镜像
   huggingface-cli download FunAudioLLM/Fun-CosyVoice3-0.5B --local-dir code/pretrained_models/Fun-CosyVoice3-0.5B
   ```
   > 权重必须落在 **`code\pretrained_models\Fun-CosyVoice3-0.5B\`**（服务按相对路径加载，目录内须含 `cosyvoice3.yaml`、`llm.pt`、`flow.pt`、`hift.pt`、`campplus.onnx`、`speech_tokenizer_v3.onnx` 等）。
6. **启动服务**：自动等待模型加载完成（约 1-3 分钟），成功后在浏览器打开 http://localhost:8188 。

> 全程各步骤幂等：依赖装过、引擎/模型已存在时会自动跳过。

## 四、日常启动 / 停止 / 状态

| 操作 | 方式 |
| ---- | ---- |
| 启动 | 双击 `启动.bat` 或 `start.bat`（模型加载约 1-3 分钟，日志在 `runtime\server.log`） |
| 停止 | 双击 `关闭.bat` 或 `stop.bat`（按进程命令行精确匹配 `app.py cosyvoice-app` 停止） |
| 状态 | 双击 `状态.bat` 或 `status.bat`（查询 /health 与 GPU 占用） |

命令行验证：

```powershell
curl http://localhost:8188/health
# => {"status":"healthy","gpu":{...,"model_loaded":true,...}}
```

## 五、默认端口 / 数据目录 / 日志

| 项 | 默认值 | 如何修改 |
|----|--------|----------|
| 端口 | 8188 | `_run_server.bat` 中 `set PORT=8188`（或环境变量 `PORT`） |
| 音色库 | `voices\` | `_run_server.bat` 中 `VOICES_DIR` |
| 输出音频 | `output\` | `_run_server.bat` 中 `OUTPUT_DIR` |
| 输入临时区 | `input\` | `_run_server.bat` 中 `INPUT_DIR` |
| 模型缓存 | `cache\modelscope` | `_run_server.bat` 中 `MODELSCOPE_CACHE` |
| 模型权重 | `code\pretrained_models\Fun-CosyVoice3-0.5B` | `_run_server.bat` 中 `MODEL_DIR` |
| 日志 | `runtime\server.log` | 服务启动日志（保留一份 server.log.old） |

> 以上目录首次运行时自动创建（已在 `.gitignore` 中忽略，不会进版本库）。

## 六、本机与目标机器可能不同的项

- Python 解释器位置（脚本自动探测 `py` 启动器 / `python`，无需手动配置）
- 项目文件夹路径（`%~dp0` 自适应，路径含空格也可）
- GPU 型号 / 显存（自动使用 CUDA；显存不足可关闭页面中无用的「已保存音色」预加载）
- 网络环境：下载依赖/模型需要访问 pypi / pytorch / modelscope / github，必要时配置镜像或代理
- 端口冲突：修改 `_run_server.bat` 的 `PORT`

## 七、常见问题排查

1. **双击脚本闪退**：手动打开 cmd 运行 `start.bat` 看报错；最常见是没装 Python 3.10-3.13 或没勾选 py launcher。
2. **依赖安装失败**：删除 `venv\.deps_installed` 标记后重新运行 `start.bat`；网络问题可重试或换 pip 镜像。
3. **引擎拉取失败**：确认 Git 已安装且能访问 github.com；也可手动按第三节第 4 步执行后重跑 `start.bat`。
4. **模型加载超时/失败**：查看 `runtime\server.log`；确认权重目录 `code\pretrained_models\Fun-CosyVoice3-0.5B` 完整（含 `cosyvoice3.yaml`）、驱动 ≥ 570、显存未被占用。
5. **手动用 python 直接跑 app.py 被拒绝启动**：服务带启动守卫，只能通过 `start.bat`/`_run_server.bat` 启动（需要 `YIJU_MANUAL_START=1` 环境变量）；这是防止后台任务误拉起的保护。
6. **首次 TTS 很慢或 ASR 报错**：首次会后台下载 Fun-ASR-Nano-2512（约 2GB）到 `cache\modelscope`，保持联网耐心等待；之后每次 3-8 秒。
7. **离线环境**：将另一台已跑通机器的 `cache\`、`code\pretrained_models\` 一并复制过来（HF_HUB_OFFLINE=1 已内置）。
8. **中文界面乱码**：服务内部为 UTF-8；控制台窗口乱码属显示编码问题，不影响功能（.bat 脚本本身为纯 ASCII）。
9. **`image.tar`（17GB）/ docker-compose.yml**：旧 Docker 方案遗留，原生版已不再需要，可删除 image.tar 释放磁盘。
10. **报 `wetext`/kaldifst 相关错**：确认用的是 Python 3.10-3.13（不是 3.14）。

## 八、验证方法（原项目自测记录，2026-08-22）

- 模型加载：Fun-CosyVoice3-0.5B，耗时约 126.9s，VRAM 约 3.21GB ✅
- TTS 合成：`/v1/audio/speech` 返回 200，生成 5.8s wav ✅
- ASR 转写：`/v1/audio/transcriptions` 约 1.3s 转写正确 ✅
- 接口：`/health`、`/v1/voices`、`/v1/audio/voices`、Web UI 全部 200 ✅

在新机器部署完成后，可运行 `tests\test_tts.ps1` / `tests\batch_tts_test.py` 做回归（见 `tests\README.md`）。
