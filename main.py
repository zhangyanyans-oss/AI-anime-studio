import os
import sys
import json
import re
import base64
import uuid
import shutil
import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Query, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from pydantic import BaseModel
from openai import OpenAI
import httpx
from dotenv import load_dotenv

# ---------- PyInstaller 打包支持 ----------
# exe 双击运行时：切到 exe 所在目录（.env、static/uploads 等持久数据都放 exe 旁边），
# 再把打包内置的前端文件补齐到本地 static（不覆盖已有的 uploads/用户数据）
if getattr(sys, "frozen", False):
    APP_DIR = os.path.dirname(sys.executable)
    os.chdir(APP_DIR)
    _bundled_static = os.path.join(sys._MEIPASS, "static")
    if os.path.isdir(_bundled_static):
        for _root, _dirs, _files in os.walk(_bundled_static):
            _rel = os.path.relpath(_root, _bundled_static)
            _dst = os.path.join("static", _rel) if _rel != "." else "static"
            os.makedirs(_dst, exist_ok=True)
            for _f in _files:
                _dstf = os.path.join(_dst, _f)
                if not os.path.exists(_dstf):
                    shutil.copy2(os.path.join(_root, _f), _dstf)

# 只读当前目录的 .env（exe 旁边 / 项目根目录），不向上级目录搜索，避免误读别处的配置
load_dotenv(os.path.join(os.getcwd(), ".env"))

app = FastAPI(title="AI Anime Studio")

# 确保图片目录存在（uploads 为持久目录，不会被自动清理；temp 仅供临时用途）
if not os.path.exists("static/temp"):
    os.makedirs("static/temp")
if not os.path.exists("static/uploads"):
    os.makedirs("static/uploads")

# ---------- 初始化客户端 ----------
# 注意：key 缺失时用占位符，保证程序能启动（OpenAI 客户端传 None 会直接抛异常）；
# 用户之后在 ⚙️ API 设置里填入真实 key，会重建客户端并写入 .env
deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY") or "not-configured",
    base_url="https://api.deepseek.com",
)

DOUBAO_API_KEY = os.getenv("ARK_API_KEY")
DOUBAO_BASE = "https://ark.cn-beijing.volces.com/api/v3"
DOUBAO_MODEL = os.getenv("DOUBAO_MODEL", "doubao-seedream-5-0-flash-260915")

AUTODL_TOKEN = os.getenv("AUTODL_TOKEN")
AUTODL_I2V_TOKEN = os.getenv("AUTODL_I2V_TOKEN", "")  # 可选：首尾帧应用(U02)独立令牌，留空则用 AUTODL_TOKEN
AUTODL_BASE = "https://autodl.art"
AUTODL_WORKFLOW = os.getenv("AUTODL_WORKFLOW_ID", "minimax_h3_zm_u24")

# ---------- 数据模型 ----------
class StoryRequest(BaseModel):
    idea: str
    image_base64: str | None = None 
    chat_history: List[Dict[str, str]] | None = None

class ImageRequest(BaseModel):
    prompt: str
    reference_urls: list[str] | None = None 
    size: str = "2048x2048"

class VideoRequest(BaseModel):
    prompt: str
    image_urls: list[str] = []
    duration: int = 5
    resolution: str = "768p"
    aspect_ratio: str = "portrait"
    workflow_id: str | None = None
    frame_mode: str = "reference"  # reference=参考图 / first_frame=首帧 / last_frame=尾帧 / first_last=首尾帧过渡
    first_frame_url: str | None = None  # 首帧模式下作为视频第一帧的图
    last_frame_url: str | None = None   # 尾帧模式下作为视频最后一帧的图

class BrainstormRequest(BaseModel):
    step: int
    messages: List[Dict[str, Any]]
    user_input: str
    image_base64: str | None = None

# ---------- 1. 头脑风暴接口 (完整版) ----------
BRAINSTORM_SYSTEM_PROMPT = """你是一个专业的动画编剧与头脑风暴助手。
你的任务是通过一步一步的提问，引导用户完善一个短篇动画的故事构思。
你需要用对话的形式，一次只问1-2个问题，不要一次性抛出所有问题。

引导流程（请循序渐进）：
1. 第一步：询问用户脑海里最核心的画面或氛围是什么（比如：一个场景、一种情绪）。
2. 第二步：引导确立主角（是谁？长什么样？有什么特征？）。
3. 第三步：挖掘冲突/目的（主角想要什么？遇到了什么阻碍？）。
4. 第四步：确立结局（高潮是什么？最后的画面定格在哪？）。

规则：
- 如果用户发了图片，请仔细观察图片内容，并基于图片中的元素（如角色长相、场景风格、色彩）进行提问和引导。
- 如果用户提供的要素还不够（比如没定下结局），请继续用简短、有启发性的问题引导。
- 当用户提供的核心要素（画面、主角、冲突、结局）基本齐全，或者用户明确表示“够了/帮我写吧”时，你需要在回复的最后加上特殊标记 `[READY_TO_GENERATE]`，并用一段连贯的文字总结出完整的故事。
"""

@app.post("/api/brainstorm")
async def brainstorm(req: BrainstormRequest):
    messages = [{"role": "system", "content": BRAINSTORM_SYSTEM_PROMPT}]
    for msg in req.messages:
        messages.append(msg)
    
    if req.user_input or req.image_base64:
        content = []
        if req.user_input:
            content.append({"type": "text", "text": req.user_input})
        if req.image_base64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{req.image_base64}"}
            })
        
        if len(content) == 1 and content[0]["type"] == "text":
            messages.append({"role": "user", "content": req.user_input})
        else:
            messages.append({"role": "user", "content": content})

    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
        )
        reply = response.choices[0].message.content
        
        if "[READY_TO_GENERATE]" in reply:
            clean_reply = reply.replace("[READY_TO_GENERATE]", "").strip()
            return {"status": "ready", "reply": clean_reply, "story_idea": clean_reply}
        else:
            return {"status": "asking", "reply": reply}
    except Exception as e:
        raise HTTPException(500, str(e))

# ---------- 2. 生成剧本、角色、分镜 (完整版) ----------
STORY_SYSTEM_PROMPT = """你是一个动画短片策划助手。用户会提供一个故事想法或聊天记录，可能还会附带一张参考图。
请结合这些信息，严格按照以下 JSON 格式输出，不要添加任何其他文字：

{
  "hook": "一句话钩子，用于吸引观众",
  "synopsis": "剧情梗概，200字以内",
  "characters": [
    {"name": "角色名", "description": "外貌、服装、风格描述", "image_prompt": "用于生图的英文/中文提示词"}
  ],
  "scenes": [
    {"name": "场景名", "description": "场景描述", "image_prompt": "用于生图的提示词"}
  ],
  "storyboards": [
    {"shot_number": 1, "description": "画面描述", "dialogue": "台词或旁白", "duration": 3, "image_prompt": "用于生成该分镜图的提示词"}
  ]
}

注意：输出必须是合法 JSON，不要用 markdown 代码块包裹。"""

@app.post("/api/generate-story")
async def generate_story(req: StoryRequest):
    try:
        user_content = []
        
        prompt_text = ""
        if req.chat_history:
            prompt_text += "以下是用户的头脑风暴对话记录，请基于此生成剧本：\n"
            for msg in req.chat_history:
                role = "用户" if msg["role"] == "user" else "AI助手"
                prompt_text += f"{role}: {msg['content']}\n"
        elif req.idea:
            prompt_text += f"故事想法：{req.idea}\n"

        if prompt_text:
            user_content.append({"type": "text", "text": prompt_text})
        if req.image_base64:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{req.image_base64}"}
            })
            
        if len(user_content) == 1 and user_content[0]["type"] == "text":
            final_content = user_content[0]["text"]
        else:
            final_content = user_content

        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": STORY_SYSTEM_PROMPT},
                {"role": "user", "content": final_content},
            ],
            response_format={"type": "json_object"},
            max_tokens=4096,
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise HTTPException(500, "DeepSeek 返回空内容，请重试")
        data = json.loads(content)
        return data
    except json.JSONDecodeError:
        raise HTTPException(500, "DeepSeek 返回的不是合法 JSON，请重试或调整提示词")
    except Exception as e:
        raise HTTPException(500, str(e))

# ---------- 3. 豆包生图 (存本地返回 URL) ----------
@app.post("/api/generate-image")
async def generate_image(req: ImageRequest):
    payload = {
        "model": DOUBAO_MODEL,
        "prompt": req.prompt,
        "size": req.size,
        "response_format": "b64_json",
        "watermark": False,
    }
    if req.reference_urls and len(req.reference_urls) > 0:
        # 豆包云端无法下载本机 127.0.0.1 URL（connection refused），与视频生成同样处理：
        # 本地图一律转 base64 data URI 直传，公网 URL 原样保留
        resolved = []
        for u in req.reference_urls:
            if u.startswith(("http://127.0.0.1", "http://localhost", "/static/")):
                resolved.append(local_url_to_data_uri(u))
            else:
                resolved.append(u)
        payload["image"] = resolved

    headers = {
        "Authorization": f"Bearer {DOUBAO_API_KEY}",
        "Content-Type": "application/json",
    }
    
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{DOUBAO_BASE}/images/generations",
            headers=headers,
            json=payload,
        )
        
        if resp.status_code != 200:
            print("\n❌ 豆包 API 报错详情:", resp.text, "\n")
            raise HTTPException(resp.status_code, f"豆包生图失败：{resp.text}")
            
        data = resp.json()
        b64_data = data["data"][0]["b64_json"]
        img_bytes = base64.b64decode(b64_data)
        
        filename = f"img_{uuid.uuid4().hex}.jpg"
        filepath = os.path.join("static/uploads", filename)
        with open(filepath, "wb") as f:
            f.write(img_bytes)

        return {"image_url": f"/static/uploads/{filename}"}

class UploadImageRequest(BaseModel):
    image_base64: str  # data URI 或纯 base64
    filename: str | None = None

# ---------- 3.5 本地图片上传（存到 uploads，返回 URL 供分镜/生视频使用） ----------
@app.post("/api/upload-image")
async def upload_image(req: UploadImageRequest):
    b64 = req.image_base64.strip()
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1] if "," in b64 else ""
    try:
        img_bytes = base64.b64decode(b64)
    except Exception:
        raise HTTPException(400, "图片数据无效")
    if len(img_bytes) < 100:
        raise HTTPException(400, "图片内容为空或损坏")
    ext = ".png" if (req.filename or "").lower().endswith(".png") else ".jpg"
    filename = f"img_{uuid.uuid4().hex}{ext}"
    filepath = os.path.join("static/uploads", filename)
    with open(filepath, "wb") as f:
        f.write(img_bytes)
    return {"image_url": f"/static/uploads/{filename}"}

# ---------- 3.6 聊天附件解析（txt/md/pdf/docx → 纯文本，供剧本对话引用） ----------
@app.post("/api/extract-file")
async def extract_file(file: UploadFile = File(...)):
    name = file.filename or "附件"
    ext = os.path.splitext(name)[1].lower()
    data = await file.read()
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(400, "文件超过 20MB，请拆分后上传")

    text = ""
    try:
        if ext in (".txt", ".md", ".csv", ".log", ".json"):
            for enc in ("utf-8", "gbk", "utf-16"):
                try:
                    text = data.decode(enc)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            if not text:
                text = data.decode("utf-8", errors="replace")
        elif ext == ".pdf":
            import io
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
            if not text.strip():
                raise HTTPException(400, "这个 PDF 提取不到文字（可能是扫描件/图片版），请换成文字版 PDF")
        elif ext == ".docx":
            import io
            import docx
            d = docx.Document(io.BytesIO(data))
            parts = [p.text for p in d.paragraphs if p.text.strip()]
            for t in d.tables:
                for row in t.rows:
                    parts.append(" | ".join(c.text.strip() for c in row.cells))
            text = "\n".join(parts)
        elif ext == ".doc":
            raise HTTPException(400, "暂不支持老版 .doc，请用 Word 另存为 .docx 后上传")
        else:
            raise HTTPException(400, f"暂不支持 {ext or '该'} 格式，支持的附件：图片 / txt / md / csv / pdf / docx")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"解析失败：{type(e).__name__}: {e}")

    if not text.strip():
        raise HTTPException(400, "文件内容为空")
    return {"name": name, "text": text, "chars": len(text)}

# ---------- 4. AutoDL 提交视频任务 ----------
def local_url_to_data_uri(url: str) -> str:
    """把本地图床 URL 转成 base64 data URI。
    AutoDL 云端无法访问本机 127.0.0.1 地址；实测 minimax_h3_zm 系列工作流
    支持 data:image/...;base64 格式的 ref_image，无需第三方图床中转。"""
    path_part = url.split("/static/")[-1].split("?")[0]
    local_path = os.path.join("static", path_part)
    if not os.path.exists(local_path):
        raise HTTPException(404, f"参考图文件不存在：{local_path}")
    ext = os.path.splitext(local_path)[1].lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(ext, "jpeg")
    with open(local_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/{mime};base64,{b64}"

@app.post("/api/generate-video")
async def generate_video(req: VideoRequest):
    # 帧模式归一化（先算好，才能决定用哪支令牌）
    frame_mode = (req.frame_mode or "reference").lower()
    if frame_mode in ("first_frame", "first"):
        frame_mode = "first"
    elif frame_mode in ("last_frame", "last"):
        frame_mode = "last"
    elif frame_mode in ("first_last", "both", "start_end"):
        frame_mode = "first_last"
    else:
        frame_mode = "reference"

    # 首尾帧应用(U02)可能使用独立令牌：配置了 AUTODL_I2V_TOKEN 且非参考模式时用它
    use_i2v_token = frame_mode != "reference" and AUTODL_I2V_TOKEN.strip()
    headers = {
        "Authorization": (AUTODL_I2V_TOKEN.strip() if use_i2v_token else AUTODL_TOKEN),
        "Content-Type": "application/json",
    }
    if use_i2v_token:
        print("🔑 使用首尾帧应用独立令牌 (AUTODL_I2V_TOKEN)")

    async with httpx.AsyncClient(timeout=60) as client:
        # minimax_h3_lightx2v 官方参数：prompt / duration / first_frame / last_frame（均必填，URL/base64）
        # 单帧模式若被平台拒绝则用同一张图补齐另一帧兜底重试（起点=终点）

        def to_val(url: str) -> str:
            return local_url_to_data_uri(url) if "/static/" in url else url

        def build_payload(put_first: bool, put_last: bool) -> dict:
            p = {"prompt": req.prompt, "duration": req.duration}
            if frame_mode == "reference":
                for i, url in enumerate(req.image_urls[:9]):
                    p[f"ref_image_{i}"] = to_val(url)
            else:
                if put_first and req.first_frame_url:
                    p["first_frame"] = to_val(req.first_frame_url)
                if put_last and req.last_frame_url:
                    p["last_frame"] = to_val(req.last_frame_url)
            # 注意：minimax_h3_zm 系列工作流不接受 resolution 参数（用工作流默认档位），
            # 传了会报 RequestParameterIsWrong，故不发送 req.resolution。
            return p

        target_workflow = req.workflow_id if req.workflow_id else AUTODL_WORKFLOW

        async def submit(pl: dict):
            # 网络瞬断容错：自动重试 2 次，仍失败才报错（避免偶发 httpx.ReadError 直接 500）
            last_exc = None
            for attempt in (1, 2, 3):
                try:
                    return await client.post(
                        f"{AUTODL_BASE}/api/v1/comfyui/comfyui_workflow/{target_workflow}",
                        headers=headers,
                        json=pl,
                    )
                except httpx.HTTPError as e:
                    last_exc = e
                    print(f"⚠️ 连接 AutoDL 异常(第{attempt}次): {type(e).__name__}，重试…")
                    await asyncio.sleep(1.5)
            raise HTTPException(503, f"连接 AutoDL 失败（网络瞬断，已重试 3 次）：{type(last_exc).__name__}，请稍后再试")

        if frame_mode == "reference":
            payload = build_payload(False, False)
        elif frame_mode == "first":
            payload = build_payload(True, False)   # 只传首帧
        elif frame_mode == "last":
            payload = build_payload(False, True)   # 只传尾帧
        else:
            payload = build_payload(True, True)    # 首尾帧过渡
        print(f"📤 提交给 AutoDL: workflow={target_workflow}, 模式={frame_mode}, 图数量={len(req.image_urls[:9])}")

        resp = await submit(payload)
        data = resp.json() if resp.status_code == 200 else {}
        # 首尾帧工作流文档要求 first_frame/last_frame 都必填：
        # 单帧模式若平台报"缺少必填参数"，用同一张分镜图补上另一帧重试（起点=终点，画面从图出发再回到图）
        if frame_mode in ("first", "last", "first_last") and data.get("code") == "RequestParameterIsWrong" \
                and "缺少必填参数" in (data.get("msg") or "") and not (req.first_frame_url and req.last_frame_url):
            missing = data.get("msg") or ""
            print(f"🔄 工作流要求首尾帧都必填（{missing}），用同一张分镜图补齐另一帧重试")
            # 用同一张图补齐空缺的另一个帧槽（起点=终点）
            if not req.last_frame_url:
                req.last_frame_url = req.first_frame_url
            if not req.first_frame_url:
                req.first_frame_url = req.last_frame_url
            payload = build_payload(True, True)
            resp = await submit(payload)
            data = resp.json() if resp.status_code == 200 else {}
        # 兜底：若参数名不被接受（报未定义参数），改用 ref_image_0 再试一次
        elif frame_mode in ("first", "last", "first_last") and data.get("code") == "RequestParameterIsWrong" \
                and "未定义的参数" in (data.get("msg") or ""):
            print("🔄 first_frame/last_frame 参数名不被接受，改用 ref_image_0 重试")
            payload = {"prompt": req.prompt, "duration": req.duration}
            for i, url in enumerate(req.image_urls[:9]):
                payload[f"ref_image_{i}"] = to_val(url)
            resp = await submit(payload)
            data = resp.json() if resp.status_code == 200 else {}
        if resp.status_code != 200:
            print(f"❌ AutoDL 报错: {resp.text}")
            raise HTTPException(resp.status_code, f"AutoDL 提交失败：{resp.text}")
        # 平台参数校验错误也返回 HTTP 200（code=RequestParameterIsWrong），需检查业务码
        if data.get("code") not in (None, "Success"):
            msg = data.get("msg") or str(data)
            if "工作流不可用" in msg and frame_mode != "reference":
                raise HTTPException(503, f"工作流 {target_workflow} 不可用：请检查工作流 ID 拼写（首尾帧正确 ID 为 minimax_h3_lightx2v）。"
                                         "所有工作流共用同一个 ComfyUI 令牌，若 ID 无误请到 autodl.art 该应用页面确认其支持 API 调用。")
            if "工作流不可用" in msg:
                rid = data.get("request_id") or ""
                raise HTTPException(503, f"工作流 {target_workflow} 当前不可用（{msg}{'，request_id: ' + rid if rid else ''}）。请确认该应用与令牌匹配；参考模式可用 zm 系列工作流。")
            raise HTTPException(500, f"AutoDL 提交失败：{msg}")
        task_id = data.get("task_id") or (data.get("data") or {}).get("task_id") or data.get("id")
        if not task_id:
            raise HTTPException(500, f"未拿到 task_id：{data}")
        return {"task_id": task_id, "raw": data}

# ---------- 5. AutoDL 查询视频结果 ----------
@app.get("/api/video-status/{task_id}")
async def video_status(task_id: str):
    headers = {"Authorization": AUTODL_TOKEN}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{AUTODL_BASE}/api/v1/comfyui/comfyui_workflow/result/{task_id}",
                headers=headers,
            )
    except httpx.HTTPError as e:
        # 网络偶发抖动：返回 UNKNOWN 让前端继续轮询，不中断流程
        print(f"⚠️ 查询任务状态网络异常（将自动重试）: {e}")
        return {"status": "UNKNOWN", "video_url": None, "raw": None}
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, f"查询失败：{resp.text}")
    data = resp.json()
    # 实测响应结构：{"code":"Success","data":{"status":"RUNNING","results":[...]}}
    inner = data.get("data") or {}
    status = str(inner.get("status") or data.get("status") or "").upper()
    video_url = None
    if status in ("SUCCESS", "COMPLETED", "SUCCEEDED"):
        results = inner.get("results") or data.get("results") or []
        for r in results:
            if isinstance(r, dict) and r.get("url"):
                video_url = r["url"]
                break
        if not video_url:
            video_url = (
                inner.get("video_url")
                or data.get("video_url")
                or data.get("output", {}).get("video_url")
            )
    return {"status": status, "video_url": video_url, "raw": data}

class SaveVideoRequest(BaseModel):
    task_id: str

@app.post("/api/save-video")
async def save_video(req: SaveVideoRequest):
    """把云端生成完成的视频下载到本地持久保存（AutoDL 返回的签名链接会过期）。"""
    headers = {"Authorization": AUTODL_TOKEN}
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.get(
            f"{AUTODL_BASE}/api/v1/comfyui/comfyui_workflow/result/{req.task_id}",
            headers=headers,
        )
        if resp.status_code != 200:
            raise HTTPException(resp.status_code, f"查询失败：{resp.text}")
        data = resp.json()
        inner = data.get("data") or {}
        status = str(inner.get("status") or "").upper()
        if status not in ("SUCCESS", "COMPLETED", "SUCCEEDED"):
            raise HTTPException(400, f"任务尚未完成（当前状态：{status}）")
        video_url = None
        for r in (inner.get("results") or []):
            if isinstance(r, dict) and r.get("url"):
                video_url = r["url"]
                break
        if not video_url:
            raise HTTPException(404, "任务完成但未找到视频文件")
        vresp = await client.get(video_url)
        if vresp.status_code != 200:
            raise HTTPException(500, "视频下载失败")
        filename = f"video_{uuid.uuid4().hex}.mp4"
        filepath = os.path.join("static/uploads", filename)
        with open(filepath, "wb") as f:
            f.write(vresp.content)
        return {"video_url": f"/static/uploads/{filename}"}

@app.get("/api/videos")
async def list_videos():
    """列出所有已保存到本地的视频（static/uploads/video_*.mp4，按时间倒序）。"""
    videos = []
    d = "static/uploads"
    if os.path.exists(d):
        files = [f for f in os.listdir(d) if f.startswith("video_") and f.endswith(".mp4")]
        files.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
        for f in files:
            p = os.path.join(d, f)
            videos.append({
                "url": f"/static/uploads/{f}",
                "name": f,
                "size_mb": round(os.path.getsize(p) / 1048576, 1),
                "mtime": int(os.path.getmtime(p) * 1000),
            })
    return {"videos": videos}

@app.delete("/api/videos/{name}")
async def delete_video(name: str):
    """删除一个已保存的本地视频文件（严格校验文件名，防止路径穿越）。"""
    if not re.fullmatch(r"video_[\w\-]+\.mp4", name):
        raise HTTPException(400, "非法文件名")
    p = os.path.join("static", "uploads", name)
    if not os.path.exists(p):
        raise HTTPException(404, "文件不存在")
    os.remove(p)
    return {"ok": True}

# ---------- 6. 清理临时文件（可选后台任务） ----------
def clean_temp_files():
    """清理超过 24 小时的临时图片文件"""
    import time
    now = time.time()
    temp_dir = "static/temp"
    if os.path.exists(temp_dir):
        for f in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, f)
            if os.path.isfile(file_path) and now - os.path.getmtime(file_path) > 86400:
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"清理失败: {e}")

# ---------- 挂载前端 ----------
# 先把 /static/* 精确映射到 static 目录（/static/uploads/xxx.jpg -> static/uploads/xxx.jpg）
# ---------- 7. 设置页面：查看/修改 API 配置 ----------
class SettingsRequest(BaseModel):
    deepseek_api_key: str | None = None
    ark_api_key: str | None = None
    doubao_model: str | None = None
    doubao_base: str | None = None
    autodl_token: str | None = None
    autodl_i2v_token: str | None = None  # 可选：首尾帧应用(U02)独立令牌
    autodl_workflow_id: str | None = None
    autodl_base: str | None = None

def _write_env_file(updates: dict):
    """更新 .env 文件：改已有键的值、追加新键，保留注释与其他行"""
    lines = []
    if os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as f:
            lines = f.readlines()
    remaining = dict(updates)
    out = []
    for line in lines:
        stripped = line.strip()
        key = stripped.split("=", 1)[0].strip() if ("=" in stripped and not stripped.startswith("#")) else None
        if key and key in remaining:
            out.append(f"{key}={remaining.pop(key)}\n")
        else:
            out.append(line)
    for k, v in remaining.items():
        out.append(f"{k}={v}\n")
    with open(".env", "w", encoding="utf-8") as f:
        f.writelines(out)

def _mask_secret(v: str) -> str:
    """密钥脱敏：只露前 4 位和后 4 位，未配置返回空串。"""
    v = (v or "").strip()
    if not v:
        return ""
    if len(v) <= 8:
        return v[:2] + "****"
    return f"{v[:4]}****{v[-4:]}"

@app.get("/api/settings")
async def get_settings():
    # 安全：API Key / Token 只返回脱敏摘要，永不明文传给前端；更换时在输入框填新值即可
    return {
        "deepseek_api_key": _mask_secret(os.getenv("DEEPSEEK_API_KEY", "")),
        "ark_api_key": _mask_secret(os.getenv("ARK_API_KEY", "")),
        "doubao_model": DOUBAO_MODEL,
        "doubao_base": DOUBAO_BASE,
        "autodl_token": _mask_secret(AUTODL_TOKEN),
        "autodl_i2v_token": _mask_secret(AUTODL_I2V_TOKEN),
        "autodl_workflow_id": AUTODL_WORKFLOW,
        "autodl_base": AUTODL_BASE,
    }

@app.post("/api/settings")
async def update_settings(req: SettingsRequest):
    global DOUBAO_MODEL, DOUBAO_BASE, AUTODL_TOKEN, AUTODL_I2V_TOKEN, AUTODL_WORKFLOW, AUTODL_BASE, deepseek_client
    updates = {}

    def set_env(env_key: str, value: str):
        updates[env_key] = value
        os.environ[env_key] = value

    if req.deepseek_api_key is not None and req.deepseek_api_key.strip():
        key = req.deepseek_api_key.strip()
        set_env("DEEPSEEK_API_KEY", key)
        deepseek_client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    if req.ark_api_key is not None and req.ark_api_key.strip():
        set_env("ARK_API_KEY", req.ark_api_key.strip())
    if req.doubao_model is not None and req.doubao_model.strip():
        DOUBAO_MODEL = req.doubao_model.strip()
        set_env("DOUBAO_MODEL", DOUBAO_MODEL)
    if req.doubao_base is not None and req.doubao_base.strip():
        DOUBAO_BASE = req.doubao_base.strip()
        set_env("DOUBAO_BASE", DOUBAO_BASE)
    if req.autodl_token is not None and req.autodl_token.strip():
        AUTODL_TOKEN = req.autodl_token.strip()
        set_env("AUTODL_TOKEN", AUTODL_TOKEN)
    if req.autodl_i2v_token is not None and req.autodl_i2v_token.strip():
        AUTODL_I2V_TOKEN = req.autodl_i2v_token.strip()
        set_env("AUTODL_I2V_TOKEN", AUTODL_I2V_TOKEN)
    if req.autodl_workflow_id is not None and req.autodl_workflow_id.strip():
        AUTODL_WORKFLOW = req.autodl_workflow_id.strip()
        set_env("AUTODL_WORKFLOW_ID", AUTODL_WORKFLOW)
    if req.autodl_base is not None and req.autodl_base.strip():
        AUTODL_BASE = req.autodl_base.strip()
        set_env("AUTODL_BASE", AUTODL_BASE)

    if updates:
        _write_env_file(updates)
        print(f"⚙️ 设置已更新并写入 .env: {list(updates.keys())}")
    return {"saved": True, "updated": list(updates.keys())}

# ---------- 前端页面响应头：禁止缓存 index.html，避免浏览器用旧页面看不到新功能 ----------
@app.middleware("http")
async def no_cache_html(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith(".html"):
        response.headers["Cache-Control"] = "no-store, must-revalidate"
    elif path.startswith("/static/uploads"):
        response.headers["Cache-Control"] = "public, max-age=604800"  # 生成物缓存 7 天
    return response

app.mount("/static", StaticFiles(directory="static"), name="assets")
# 再把根路径映射到前端页面（保持 html=True 以便访问 / 时自动返回 index.html）
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # 启动时清理一次旧文件
    clean_temp_files()
    uvicorn.run(app, host="0.0.0.0", port=8000)