# 🎬 AI 动画工作台（AI Anime Studio）

> 从一个点子到分镜图、再到视频成片的**一站式网页工具**。
> 剧本创作 → 分镜图 → 图生视频 → 本地保存，全程在一个页面里完成。

## ✨ 功能一览

- 💡 **AI 头脑风暴**：输入点子，多轮对话聊成完整创意
- 📖 **剧本生成**：开场钩子、剧情大纲、角色、场景、道具、分镜脚本一次到位
- 🎨 **分镜图生成**：每条分镜一键出图，支持多张备选、追加生成、本地图片上传
- 🧰 **道具参考图**：对剧情有意义的道具单独出设定图，分镜图自动参考，道具长相全片一致
- 🗺️ **调度参考图**：每条分镜可生成「📐 俯视调度图」（上帝视角平面走位 + 机位）和「🏃 轨迹调度图」（角色运动路径 + 镜头轨迹），生成视频时勾选即可作为额外参考图传给 AI，让运镜与走位更可控
- 🏁 **图生视频**，两种模式可选：
  - 🧩 **参考图模式**：最多 9 张图做角色/场景参考，可勾选"严格一致性约束"锁定画风与服装
  - 🎞️ **首帧 / 尾帧 / 首尾帧**：指定视频第一帧、最后一帧，AI 补全中间过渡（工作流 `minimax_h3_lightx2v`）
- ⏳ **进度条内嵌分镜**：视频生成进度直接显示在对应分镜下方，完成后原地播放
- 🖼️ **风格库**：37 种画风的参考缩略图（同一主体换画风，便于横向比对），点卡片即可复制风格描述
- 💾 **全量持久化**：项目、图片、视频全部本地保存，刷新/关机不丢
- 🔐 **API 设置页**：密钥脱敏显示、留空即保持不变、写入本地 `.env`
- 📦 **一键打包 exe**：PyInstaller 单文件分发，双击即用

## 🛠️ 技术栈

| 层 | 技术 |
|---|---|
| 后端 | Python + FastAPI + httpx（异步） |
| 前端 | 原生 HTML/JS 单文件，零框架零构建，localStorage 持久化 |
| 剧本 | DeepSeek（对话式生成） |
| 分镜图 | 火山方舟 · 豆包 seedream（支持竖屏 1080×1920 等比例） |
| 视频 | AutoDL.Art · MiniMax-H3 工作流 API（云端 GPU 渲染，按秒计费） |
| 打包 | PyInstaller（onefile） |

## 🚀 快速开始

### 方式一：源码运行

```bash
# 1. 创建虚拟环境并安装依赖
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. 配置密钥（复制模板后填入自己的 key）
copy .env.example .env

# 3. 启动
run.bat
# 或
uvicorn main:app --reload
```

浏览器打开 http://127.0.0.1:8000 即可使用。

### 方式二：exe 直接运行（推荐给非技术用户）

用 PyInstaller 打包成单文件：

```bash
pip install pyinstaller
pyinstaller --onefile --name "AI动画工作台" --add-data "static/index.html;static" run_app.py
```

- 双击 `dist/AI动画工作台.exe` → 浏览器自动打开
- 关闭控制台窗口即退出服务
- **`.env` 不是必需的**：没有也能启动，首次打开页面在 ⚙️ API 设置里填入密钥即可，会自动保存到 exe 旁边

## 🔑 需要准备的密钥

| 服务 | 用途 | 获取方式 |
|---|---|---|
| DeepSeek | 剧本创作 | platform.deepseek.com |
| 火山方舟（ARK） | 分镜图（豆包 seedream） | console.volcengine.com/ark |
| AutoDL.Art 令牌 | 视频生成（分组选 **ComfyUI**） | autodl.art/large-model/tokens |

## 💰 成本参考

视频生成按秒计费，白天约 ¥0.02~0.03/秒，**夜间 00:00–8:00 半价**；5 秒 480p 一条约 ¥0.10。图片生成约 ¥0.01~0.03/张。生成失败不扣费。

## 📁 项目结构

```
AI-anime-studio/
├── main.py              # FastAPI 后端（剧本/生图/调度图提示词/生视频/设置/上传）
├── run_app.py           # exe 打包入口（自动选端口 + 打开浏览器）
├── run.bat              # 开发模式启动脚本
├── requirements.txt     # Python 依赖
├── static/index.html    # 前端单文件（全部 UI 逻辑）
├── static/style-previews/ # 风格库缩略图（37 张，1200×675）
├── make_style_previews.py # 风格库配图生成（豆包 seedream，同一主体换 37 种画风）
├── static/uploads/      # 运行时生成的图片/视频（自动创建，不入库）
└── .env                 # 密钥配置（不入库，模板见 .env.example）
```

## 💡 踩坑提醒

- 🔤 工作流 ID 必须**逐字母核对**：`minimax_h3_lightx2v`（light**x**2v），差一个字母只报"工作流不可用"，非常迷惑
- 🖼️ 分镜图比例要与视频比例匹配（默认竖屏 portrait），否则会被裁切变形
- 🔗 云端返回的视频链接有效期很短，程序会自动下载到 `static/uploads/`，无需手动抢救
- 🧩 参考图经 base64 直传云端，不依赖图床，不怕外链失效

## 📄 License

MIT
