<div align="center">

# 🎙️ 一句话克隆声音 — CosyVoice 声音克隆 / TTS 服务

> ⭐ **喜欢这个项目？请先点个 Star ⭐ 支持一下，让更多人看到！**

<!-- 自动徽章（公开后自动显示星星数/语言/许可证），占位写法，推送前按仓库替换：
![GitHub stars](https://img.shields.io/github/stars/yishui111/yijuhuakelongshengyin.svg?style=flat-square&color=orange)
![GitHub forks](https://img.shields.io/github/forks/yishui111/yijuhuakelongshengyin.svg?style=flat-square)
![GitHub repo size](https://img.shields.io/github/repo-size/yishui111/yijuhuakelongshengyin.svg?style=flat-square)
-->

**录一句 3-30 秒的话 → 克隆出这个音色 → 输入任意文本让它开口说话。**

一个自带 Web 控制台的零样本声音克隆 + TTS 服务：基于阿里 **Fun-CosyVoice3-0.5B**，**原生运行在 Windows + NVIDIA GPU 上，不需要 Docker**，支持 OpenAI 兼容 TTS API。

</div>

---

## ✨ 项目简介

想用自己的声音（或任何人的授权声音）朗读任意文本？本项目把「零样本声音克隆 + 语音合成」封装成一个开箱即用的本地服务：

- **Web 控制台**（中文界面，内置在服务里）：上传参考音频 → 一键克隆音色 → 输入文字合成试听/下载
- **OpenAI 兼容 API**：`POST /v1/audio/speech` 一条 curl 即可接入任意 OpenAI 语音客户端
- **长文本分段合成**：长文本自动按句号逐句切分、一句一句合成（比整体合成更像、更稳），每段单独下载或一键打包 zip
- **合成质量自动保障**：合成后用 ASR 检测「参考文本复述泄漏 / 丢字 / 乱码」，异常自动重试
- 适合：个人语音助手、有声内容制作、声音类应用开发、语音合成学习研究

## 🎯 主要功能

- 🎤 **零样本声音克隆**：上传 3-30 秒参考音频 + 参考文本，自动创建自定义音色（页面可试听）
- 🔊 **TTS 合成**（OpenAI 兼容 `/v1/audio/speech`，支持 speed/stream、自定义与预置音色）
- 📄 **长文本分段合成**（`/api/tts/batch`）：按「。」自动逐句合成 + 实时进度 + 一键 zip 下载
- 🧪 **合成质量自检**：Fun-ASR-Nano 检测复述泄漏与丢字，自动重试（最多 3 次）
- 🗣 **ASR 转写**（`/v1/audio/transcriptions`）：语音转文字，附带「音色参考文本一键重转写」纠错
- 🌐 **多语言**：中、英、日、韩、德、西、法、意、俄 + 18+ 中文方言（引擎能力）
- 🧠 **MCP 服务**（可选，`code/mcp_server.py`）：供 AI 助手通过 Model Context Protocol 调用克隆/合成

## 🗂️ 目录结构

```
yijuhuakelongshengyin/
├── code/                       # 自研服务代码（本仓库核心）
│   ├── app.py                  #   主服务：FastAPI + 内嵌 Web 控制台（端口 8188）
│   ├── model.py                #   Fun-ASR 远程代码（语音识别/质量自检必需）
│   ├── mcp_server.py           #   可选 MCP 服务（实验性）
│   ├── cosyvoice/              #   ⚠️ 第三方引擎（不入库，部署时自动下载）
│   ├── third_party/            #   ⚠️ Matcha-TTS（不入库，部署时自动下载）
│   └── pretrained_models/      #   ⚠️ 模型权重（不入库，见下方「大件下载」）
├── tests/                      # 功能测试脚本（test_tts.ps1 / batch_tts_test.py / tts_native_test.py）
├── tools/
│   ├── download_engine.bat     #   一键拉取 CosyVoice 引擎 + Matcha-TTS（首次）
│   ├── download_model.bat      #   一键下载 Fun-CosyVoice3-0.5B 权重（首次）
│   └── extract_image.py        #   历史工具：从旧 Docker image.tar 提取文件（已不需要）
├── requirements.txt            # Python 依赖锁定（torch 2.11+cu128 等，Windows 原生）
├── start.bat / 启动.bat        # 一键启动（含首次自动建 venv、装依赖、拉引擎/模型引导）
├── stop.bat / 关闭.bat         # 一键停止
├── status.bat / 状态.bat       # 查看服务/GPU 状态
├── _run_server.bat             # 服务进程启动器（由 start.bat 调用，勿直接双击）
├── docker-compose.yml          # ⚠️ 旧 Docker 方案遗留（存档，已不使用）
└── README.md / DEPLOY.md / 使用说明.md
```

> 💡 本仓库只包含**自研代码 / 脚本 / 配置 / 文档**。
> 模型权重、第三方引擎源码、真人声音素材、运行缓存等**大文件与敏感内容不随仓库分发**，见下方「大件资源下载」与 `DEPLOY.md`。

## 🚀 快速开始（拉到新电脑即可部署）

### 环境要求

- 操作系统：Windows 10/11 x64（中文/英文均可）
- GPU：NVIDIA 显卡，显存 ≥ 8GB（推荐 16GB）；驱动 ≥ 570（CUDA 12.8）
- 软件：Python 3.10–3.13（勾选 py launcher）、Git for Windows
- 磁盘：≥ 25GB 空闲；需要联网（首次下载依赖/模型）

### 1. 克隆

```bash
git clone https://github.com/yishui111/yijuhuakelongshengyin.git
cd yijuhuakelongshengyin
```

### 2. 一键启动（推荐）

双击 **`start.bat`**（或中文习惯用 **`启动.bat`**）。首次运行会自动完成：

1. 创建 `venv` 并安装依赖（torch ~3GB，需 5-15 分钟）；
2. 自动拉取第三方引擎代码：`tools\download_engine.bat`（CosyVoice 引擎 + Matcha-TTS）；
3. 提示下载模型权重：`tools\download_model.bat`（Fun-CosyVoice3-0.5B，约 7GB）；
4. 启动服务并等待模型加载（约 1-3 分钟），自动打开浏览器。

### 3. 验证

浏览器访问 <http://localhost:8188>（API 文档 <http://localhost:8188/docs>），页面可克隆音色并合成试听；也可用命令行验证：

```bash
curl http://localhost:8188/health
# => {"status":"healthy", ...}
```

> 详细的手动部署步骤、常见问题排查见 `DEPLOY.md`；页面功能使用见 `使用说明.md`。

### Windows 启停脚本

| 操作 | 英文脚本 | 中文脚本 |
| ---- | ---- | ---- |
| 启动 | `start.bat` | `启动.bat` |
| 停止 | `stop.bat` | `关闭.bat` |
| 状态 | `status.bat` | `状态.bat` |

## 📥 大件资源下载（模型 / 引擎 / 依赖 — 均不入库）

| 资源 | 用途 | 下载地址 / 获取方式 |
| ---- | ---- | ---- |
| Fun-CosyVoice3-0.5B 权重（~7GB） | 克隆/合成主模型 | ModelScope：`FunAudioLLM/Fun-CosyVoice3-0.5B`（[链接](https://modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B)）；HuggingFace 镜像同名。运行 `tools\download_model.bat` |
| CosyVoice 引擎源码（cosyvoice 包） | 运行时引擎（Apache-2.0） | GitHub：`FunAudioLLM/CosyVoice`。运行 `tools\download_engine.bat` 自动拉取到 `code\cosyvoice` |
| Matcha-TTS（数 MB，MIT） | 引擎配套（flow 匹配网络） | GitHub：`shivammehta25/Matcha-TTS`。同上自动装到 `code\third_party\Matcha-TTS` |
| Fun-ASR-Nano-2512（~2GB） | 合成质量自检 / ASR 转写 | 首次启动后台自动下载到 `cache\modelscope`（ModelScope：`FunAudioLLM/Fun-ASR-Nano-2512`） |
| Python 依赖（torch 2.11+cu128 等 ~3GB） | 运行环境 | `pip install -r requirements.txt`（`start.bat` 首次自动执行） |

## 🛠️ 本地开发 & 提交

```bash
# 修改 code\app.py 后重启服务生效；改完按 tests\README.md 跑一遍回归
git add .
git commit -m "feat: xxx"
git push origin main   # 推送前把 README/DEPLOY 中的 yishui111 替换为真实用户名
```

## ❓ 常见问题（FAQ）

- **Q：双击 start.bat 闪退 / 报找不到 Python？** A：需安装 Python 3.10-3.13 并勾选 "py launcher"（https://www.python.org/downloads/）；3.14 没有 kaldifst/wetext 轮子不可用。手动打开 cmd 运行 `start.bat` 可看到具体报错。
- **Q：模型加载要多久？** A：首次 1-3 分钟属正常（加载约 127s、占用显存约 3.2GB），期间 `/health` 可能连不上；之后每次启动约 1 分钟。
- **Q：首次 TTS 很慢？** A：首次会额外下载/加载 Fun-ASR 检测模型（后台预热），之后每次合成约 3-8 秒。
- **Q：长文本整体合成效果差 / 慢？** A：用页面「📄 长文本分段合成」按句逐句合成，每句可单独下载或一键打包 zip。
- **Q：合成结果带参考文本（复述）？** A：服务已内置泄漏检测 + 自动重试（zero_shot ↔ cross_lingual 兜底），无需手动处理。
- **Q：怎么改端口？** A：编辑 `_run_server.bat` 中 `set PORT=8188`（也支持环境变量 `PORT`）。
- **Q：报 torch 相关 CUDA 错误？** A：确认 NVIDIA 驱动 ≥ 570（torch 2.11 为 cu128 构建）；本机非 NVIDIA 显卡无法推理。
- **Q：image.tar / docker-compose.yml 是干嘛的？** A：早期 Docker 方案遗留（image.tar 约 17GB 已可删除），本项目已迁移为 Windows 原生运行，不再需要 Docker。

## ⚠️ 注意事项

- 声音克隆涉及个人声音权益，**请仅克隆你本人或已获授权的声音**，遵守当地法律法规与平台规范；
- 敏感信息（密钥、token、账号密码）一律放环境变量或 `.env`，禁止提交到仓库；
- 仓库不含模型权重、第三方引擎源码、真人声音素材与运行产物，首次部署需按 `DEPLOY.md` 下载；
- 本仓库仅供学习交流使用。

## 📄 许可证

本项目自研代码按 **MIT License** 提供。随项目运行的第三方组件遵循其各自许可：
CosyVoice 引擎（Apache-2.0，来自 FunAudioLLM/CosyVoice）、Matcha-TTS（MIT）、
Fun-CosyVoice3-0.5B / Fun-ASR-Nano 模型遵循其模型发布页的许可条款。

## 🙏 支持与致谢

- 模型与引擎：[FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)（Fun-CosyVoice3-0.5B）、[Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS)、Fun-ASR-Nano
- 如果这个项目帮到了你，**请点亮右上角的 ⭐ Star**，你的支持是我持续更新的最大动力！
