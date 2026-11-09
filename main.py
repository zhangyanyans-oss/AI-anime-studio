import os
import sys
import json
import re
import io
import base64
import uuid
import shutil
import asyncio
import zipfile
from urllib.parse import quote
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Query, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from pydantic import BaseModel
from openai import OpenAI
import httpx
from dotenv import load_dotenv
import au_lexicon

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
                # uploads/temp 是用户持久数据，绝不覆盖；其余（index.html、
                # style-previews 等）是程序文件，随新版本 exe 覆盖更新
                _parts = {p.lower() for p in os.path.normpath(_dstf).split(os.sep)}
                if _parts & {"uploads", "temp"}:
                    continue
                try:
                    shutil.copy2(os.path.join(_root, _f), _dstf)
                except OSError as _e:
                    # 个别文件被占用/权限不足时不中断启动（下次启动会再尝试覆盖）
                    print(f"⚠ 更新 {_dstf} 失败（跳过）: {_e}")

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

def deepseek_friendly_error(e: Exception) -> str:
    """把 DeepSeek 调用异常翻译成可操作的人话（方便在别人机器上排查）。"""
    raw = str(e)
    low = raw.lower()
    if getattr(deepseek_client, "api_key", "") in ("not-configured", None, ""):
        return ("未配置 DeepSeek API Key，无法生成剧本/头脑风暴。请在程序(exe)同目录创建 .env 文件，"
                "写入一行：DEEPSEEK_API_KEY=sk-你的key；或打开页面右上角 ⚙️ API 设置直接粘贴 Key 并保存。")
    if "401" in raw or "authentication" in low or ("invalid" in low and "key" in low):
        return "DeepSeek API Key 无效或已过期，请检查 .env 里的 DEEPSEEK_API_KEY 是否正确。"
    if "402" in raw or "insufficient balance" in low:
        return "DeepSeek 账户余额不足，请到 platform.deepseek.com 充值后重试。"
    if "429" in raw:
        return "DeepSeek 请求过于频繁，请稍等几秒再试。"
    if isinstance(e, (ConnectionError, TimeoutError)) or "connection" in low or "timeout" in low or "getaddrinfo" in low or "ssl" in low:
        return ("无法连接 DeepSeek 服务器（api.deepseek.com）。请确认这台电脑可以上网；"
                "公司/校园网络或防火墙可能拦截了该域名，可尝试换网络。")
    return f"DeepSeek 调用失败：{raw[:200]}"

# ---------- 模型 JSON 输出的健壮解析 ----------
def extract_json_text(content: str) -> str:
    """剥掉 DeepSeek 偶尔夹带的 markdown 代码块围栏与前后杂文本，只留 JSON 本体。"""
    text = content.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
    if m:
        text = m.group(1).strip()
    else:
        # 没有围栏时，截取第一个 { 到最后一个 } 之间的部分
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start:end + 1]
    return text

def repair_json_text(text: str) -> str:
    """轻量修复常见 JSON 瑕疵：全角/智能引号、尾逗号、中文冒号。"""
    text = text.replace("\u201c", '"').replace("\u201d", '"')   # "" -> "
    text = text.replace("\u2018", "'").replace("\u2019", "'")   # '' -> '
    text = text.replace("\uff1a", ":").replace("\uff0c", ",")   # ：, ->
    text = re.sub(r",\s*([}\]])", r"\1", text)                  # 去尾逗号
    return text

def parse_model_json(content: str, finish_reason: str | None = None) -> Any:
    """尽力解析模型返回的 JSON；失败抛 ValueError，错误信息带具体原因。"""
    if finish_reason == "length":
        raise ValueError("输出长度超出上限被截断")
    text = extract_json_text(content)
    candidates = [text]
    repaired = repair_json_text(text)
    if repaired != text:
        candidates.append(repaired)
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"无法从模型输出中解析出 JSON（输出开头：{content[:80]!r}）")

def deepseek_json(response) -> Any:
    """从 DeepSeek 响应对象中解析 JSON，失败抛 ValueError（带原因）。"""
    choice = response.choices[0]
    return parse_model_json(choice.message.content or "", getattr(choice, "finish_reason", None))

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
    reference_meta: list[Dict[str, Any]] | None = None  # 与 reference_urls 一一对应：{kind, name, description}
    context: str | None = None  # 生成上下文（分镜/场景描述），用于筛选相关参考
    size: str = "2048x2048"
    diagram_brief: str | None = None  # 调度图纵深解读（人物近/中/远景层次），由 /api/staging-brief 生成
    plot_data: dict | None = None  # 调度图精准坐标数据（/api/render-diagram 返回的 plot），翻译成自然语言注入生图

class VideoRequest(BaseModel):
    prompt: str
    image_urls: list[str] = []
    duration: int = 5
    resolution: str = "768p"
    workflow_id: str | None = None
    frame_mode: str = "reference"  # reference=参考图 / first_frame=首帧 / last_frame=尾帧 / first_last=首尾帧过渡
    first_frame_url: str | None = None  # 首帧模式下作为视频第一帧的图
    last_frame_url: str | None = None   # 尾帧模式下作为视频最后一帧的图
    image_aspect: float | None = None   # 首张参考图的宽/高比（前端计算传入），用于自动选横竖屏档位
    plot_data: dict | None = None       # 调度图坐标数据（轨迹图优先），翻译成自然语言运动轨迹指令注入视频提示词

class BrainstormRequest(BaseModel):
    step: int
    messages: List[Dict[str, Any]]
    user_input: str
    image_base64: str | None = None

class RegeneratePromptRequest(BaseModel):
    type: str  # character / scene / storyboard
    name: str | None = None
    description: str | None = None
    dialogue: str | None = None            # 分镜用
    duration: int = 3                      # 分镜用
    style_bible: str | None = None         # 全片画风圣经
    synopsis: str | None = None            # 剧情梗概
    characters: List[Dict[str, Any]] | None = None  # [{name, image_prompt}] 角色外观锚点
    old_prompt: str | None = None          # 旧提示词（在其基础上重写）
    feedback: str | None = None            # 用户的新要求（最高优先级）

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
    if getattr(deepseek_client, "api_key", "") in ("not-configured", None, ""):
        raise HTTPException(400, "未配置 DeepSeek API Key。请在程序(exe)同目录创建 .env 文件写入 DEEPSEEK_API_KEY=sk-你的key，"
                                 "或打开页面右上角 ⚙️ API 设置粘贴 Key 并保存后重试。")
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
        raise HTTPException(500, deepseek_friendly_error(e))

# ---------- 2. 生成剧本、角色、分镜 (工业级提示词模板 v2) ----------
STORY_SYSTEM_PROMPT = """你是一位资深的 AI 动画短片导演兼提示词工程师，精通为文生图模型（豆包 seedream）和图生视频模型（MiniMax-H3）撰写工业级提示词。
用户会提供一个故事想法或头脑风暴对话记录，可能附带一张参考图（若附带，请沿用其画风、色彩与角色形象，并提炼进 style_bible）。

【输出格式】严格输出以下 JSON（键名不得改动，不要 markdown 代码块，不要任何 JSON 之外的文字）：
{
  "style_bible": "全片统一画风圣经：一段 50-80 字的固定画风描述（媒介类型+画风流派+主色调+光影基调+画质关键词）。后续每条分镜 image_prompt 都必须以它开头逐字复用，这是全片画风一致性的核心。",
  "hook": "一句话钩子，用于吸引观众",
  "synopsis": "剧情梗概，200字以内",
  "characters": [
    {"name": "角色名",
     "description": "角色设定：年龄、身份、性格，一句话",
     "image_prompt": "角色设定图提示词，固定写法：style_bible 原文 + 「单人全身立绘、完整入画（头顶到脚踝全部可见）、正面或四分之三侧面、自然站姿目视镜头、简洁纯色浅背景、柔和均匀打光」+ 逐一写死可复用的外观锚点：发型发色、瞳色、五官特征、体型年龄、服装款式与颜色、随身标志物。60-120 字。此图将作为该角色在所有分镜中的参考图，因此禁止出现场景、动作戏、其他人物。"}
  ],
  "scenes": [
    {"name": "场景名",
     "description": "场景的功能与氛围，一句话",
     "image_prompt": "场景设定图提示词：style_bible 原文 + 大远景空镜（画面中无人物）+ 空间布局与 2-3 个标志性陈设 + 时间天气 + 光线方向。60-120 字。此图将作为场景参考图。"}
  ],
  "props": [
    {"name": "道具名",
     "description": "道具在剧情中的作用与意义，一句话",
     "image_prompt": "道具设定图提示词，固定写法：style_bible 原文 + 「单件道具特写陈列图，画面中无人物，浅色纯净背景，柔和均匀布光，可附比例参照」+ 材质、颜色、形态结构、新旧磨损细节、标志性特征。40-80 字。此图将作为该道具在分镜中的参考图，禁止出现场景环境或无关物体堆砌。"}
  ],
  "storyboards": [
    {"shot_number": 1,
     "description": "本镜头的剧情作用，一句话",
     "dialogue": "该镜头的台词或旁白，没有则输出空字符串",
     "duration": 3,
     "image_prompt": "分镜图提示词（文生图用），必须按结构化模板撰写：单行输出，小节用【】标题依次排列，每个小节必须写，禁止漏节。【画面内容】在(地点)，(谁)正在(做什么)，画面正在发生(具体事件)，角色逐字复用其 image_prompt 的外貌服装锚点，写明表情与动作瞬间；【镜头细节】焦段(如35mm/85mm)、构图(主体居中/对称/前景遮挡)、景别(远/全/中/近/特)、机位(固定/低角度/俯拍/仰拍)；【美术风格】style_bible 原文；【光源】光源类型(顶光/逆光/侧光/蝴蝶光/伦勃朗光/硬光/柔光/自然光等)及光如何影响场景；【氛围】观众的情绪感受(开心/孤独/紧张/充满希望等)；【影调】冷暖色调、饱和度、明暗关系；【约束】正向表达：画面无文字水印分格、动作自然符合物理规律、本镜头需锁定的要点。上下文不足以确定的内容写【待补：说明】占位留给创作者修改，禁止编造关键设定。",
     "video_prompt": "图生视频提示词，单行输出：【动作时序】首先…然后…最后…(总时长=duration秒)；【镜头运动】推/拉/摇/移/环绕/跟随/固定机位；【声音】仅当剧情有明确声音需求时注明，没有就省略此节。禁止引入分镜图中不存在的新人物/物体/场景，禁止“突然”“闪现”“切换到”等跳变词，不复述静态外观。"}
  ]
}

【导演规则】
1. storyboards 数量 8-16 个，按剧情节奏编排；第一个镜头必须能在 3 秒内抓住眼球（悬念/冲击画面/反差）。
2. 景别要有剪辑感：相邻镜头避免连续使用相同景别，重点情绪用特写强调，转场处可用大远景。
3. 同一场景内的相邻镜头，时间、天气、光线必须连续一致；跨场景才允许变化。
4. duration 只能取 2 到 5 的整数：静态对话 2-3 秒，动作镜头 3-5 秒。
5. dialogue 短而口语化，单镜头不超过两句；没有台词就留空字符串，不要硬凑。
6. 所有分镜 image_prompt 中同一角色的外观描述，必须与该角色 image_prompt 逐字相同——这是参考图机制下保持角色一致性的关键。
7. 全片画风、色调必须统一，由 style_bible 锁定；不得中途变换风格。
8. 素材引用（参考图/音频/视频）与全局设定（资产库）是创作者的输入，不由你生成；不要虚构素材，相关需求写进【约束】或【待补】。
9. 每条分镜 image_prompt 必须包含全部【】小节且单行输出；不确定的小节写【待补：说明】，让创作者修改后重写。
10. 输出必须是可直接 json.loads 的合法 JSON：双引号、无注释、无尾逗号，字符串内不要出现真实换行。
11. props 数量 0-6 个：只挑对剧情有实际作用的道具（信物、钥匙、武器、信件等能推动剧情或值得特写的），普通背景陈设不要列入；没有就输出空数组。"""

@app.post("/api/generate-story")
async def generate_story(req: StoryRequest):
    if getattr(deepseek_client, "api_key", "") in ("not-configured", None, ""):
        raise HTTPException(400, "未配置 DeepSeek API Key。请在程序(exe)同目录创建 .env 文件写入 DEEPSEEK_API_KEY=sk-你的key，"
                                 "或打开页面右上角 ⚙️ API 设置粘贴 Key 并保存后重试。")
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

        last_err = "未知错误"
        for attempt in range(2):  # 解析失败自动重试 1 次
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": STORY_SYSTEM_PROMPT},
                    {"role": "user", "content": final_content},
                ],
                response_format={"type": "json_object"},
                max_tokens=16384,
            )
            choice = response.choices[0]
            content = choice.message.content or ""
            if not content.strip():
                last_err = "DeepSeek 返回空内容"
                print(f"[generate-story] 第 {attempt + 1} 次输出为空，{'重试' if attempt == 0 else '放弃'}")
                continue
            try:
                return parse_model_json(content, getattr(choice, "finish_reason", None))
            except ValueError as e:
                last_err = str(e)
                print(f"[generate-story] 第 {attempt + 1} 次输出解析失败：{last_err}")
        raise HTTPException(
            500,
            f"DeepSeek 连续 2 次返回的内容都无法解析成 JSON（{last_err}）。"
            "请再试一次；若反复出现，多半是剧本太长导致输出被截断，可缩短故事想法后重试。"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, deepseek_friendly_error(e))

# ---------- 2.5 单条提示词重生成（人物 / 场景 / 分镜） ----------
REGEN_SYSTEM_PROMPT = """你是 AI 动画工作台的提示词重写引擎，负责把一条不够精细的生图提示词重写成工业级提示词。

【画风圣经】用户会提供全片统一的画风描述（style_bible），重写后的 image_prompt 必须以它的原文开头，保证全片画风一致。

【四类提示词的固定规范】
1. 角色（character）：角色设定图。style_bible 原文 + 「单人全身立绘、完整入画（头顶到脚踝全部可见）、正面或四分之三侧面、自然站姿目视镜头、简洁纯色浅背景、柔和均匀打光」+ 逐一写死可复用的外观锚点：发型发色、瞳色、五官特征、体型年龄、服装款式与颜色、随身标志物。60-120 字。禁止出现场景、动作戏、其他人物。
2. 场景（scene）：场景设定空镜。style_bible 原文 + 大远景空镜（画面中无人物）+ 空间布局与 2-3 个标志性陈设 + 时间天气 + 光线方向。60-120 字。严禁出现任何人物或人影。
3. 道具（prop）：道具设定图。style_bible 原文 + 「单件道具特写陈列图，画面中无人物，浅色纯净背景，柔和均匀布光，可附比例参照」+ 材质、颜色、形态结构、新旧磨损细节、标志性特征。40-80 字。禁止出现场景环境或无关物体堆砌。
4. 分镜（storyboard）：结构化模板，单行输出，小节用【】标题依次排列，每个小节必须写：【画面内容】在(地点)，(谁)正在(做什么)，画面正在发生(具体事件)，出现的角色逐字复用其角色锚点的外貌服装描述，写明表情与动作瞬间；【镜头细节】焦段(如35mm/85mm)、构图(主体居中/对称/前景遮挡)、景别(远/全/中/近/特)、机位(固定/低角度/俯拍/仰拍)；【美术风格】style_bible 原文；【光源】光源类型(顶光/逆光/侧光/蝴蝶光/伦勃朗光/硬光/柔光/自然光等)及光如何影响场景；【氛围】观众的情绪感受；【影调】冷暖色调、饱和度、明暗关系；【约束】正向表达：无文字水印分格、动作自然符合物理规律、本镜头需锁定的要点。上下文不足写【待补：说明】，禁止编造。

【硬性禁止】画面出现任何文字、字幕、水印、边框、分格、多格拼图；禁止抽象词（如“幸福”“回忆”“命运”）；禁止把台词写进画面。

【video_prompt 规范】仅分镜类型需要输出，单行输出：【动作时序】首先…然后…最后…(总时长=该镜头持续秒数)；【镜头运动】推/拉/摇/移/环绕/跟随/固定机位；【声音】仅当有明确声音需求时注明，没有就省略。画面起点即该分镜图。硬性禁止：引入分镜图中不存在的新人物/物体/场景；禁止“突然”“闪现”“切换到”等跳变词；不要复述静态外观。

【用户修改要求】旧提示词是用户当前工作框内的内容，是最高优先级的内容来源：其中用户亲笔写下的具体细节（人物年龄、体型、五官、服装、随身物、动作、表情、地点、道具、光源、氛围、风格等）必须逐字保留，严禁替换、省略、泛化或"换种说法"。重写只允许做三件事：①把内容套进对应类型的结构化模板并分节；②补全模板要求但用户没写的维度（不确定的写【待补：说明】，禁止编造）；③语言规范化。若角色锚点与用户描述冲突，一律以用户描述为准；若旧提示词很短且明显是一句修改指令（如"换成雨夜"）而非完整提示词，将其整体视为用户的新要求，结合上下文重写出完整的结构化提示词。
【追加的指令要落实】用户在框内追加或修改的指令式短句（如"空镜头""不要人物""去掉XX""换成雨夜""加一面镜子"）是用户新要求：必须把它落实为规范中对应位置的具体描述（例如"空镜头"落实为"大远景空镜（画面中无人物）"，并保证全文没有任何人物痕迹），而不是把原句原样缀在提示词结尾；落实完成后提示词里不得残留指令原文。判断依据：该短句与提示词其余部分风格割裂、位置突兀（多为结尾），或描述的是"要改什么"而不是"画面里有什么"。

【待补占位规则】若旧提示词含有【待补：xx】，而用户的新要求没有给出对应信息，则原样保留该【待补：xx】（可以写得更具体），不要替用户拍板编造；用户在新要求里明确了的内容，必须替换掉对应的【待补】。
【素材与资产】素材引用（参考图/音频/视频）与全局设定（资产库）是创作者的输入，不由你生成、不要虚构；相关需求写进【约束】或【待补】。

【输出】严格输出 JSON：{"image_prompt": "...", "video_prompt": "..."}。非分镜类型 video_prompt 输出空字符串。不要 markdown 代码块，不要任何 JSON 之外的文字。"""

@app.post("/api/regenerate-prompt")
async def regenerate_prompt(req: RegeneratePromptRequest):
    if getattr(deepseek_client, "api_key", "") in ("not-configured", None, ""):
        raise HTTPException(400, "未配置 DeepSeek API Key，无法重新生成提示词。")
    t = (req.type or "").lower()
    if t not in ("character", "scene", "prop", "storyboard"):
        raise HTTPException(400, "type 必须是 character / scene / prop / storyboard")

    type_label = {"character": "角色设定图", "scene": "场景设定空镜", "prop": "道具设定图", "storyboard": "分镜图"}[t]
    user_lines = []
    if req.style_bible:
        user_lines.append(f"画风圣经（重写后必须以其原文开头）：{req.style_bible}")
    if req.synopsis:
        user_lines.append(f"剧情梗概：{req.synopsis}")
    if t == "storyboard" and req.characters:
        anchors = "；".join(
            f"{str(c.get('name') or '')}：{str(c.get('image_prompt') or '')}" for c in req.characters
        )
        user_lines.append(f"角色外观锚点（用户提示词里没写外貌细节时才参考复用；若与用户当前版本的提示词冲突，以用户提示词为准）：{anchors}")
    user_lines.append(f"要重写的提示词类型：{type_label}")
    if req.name:
        user_lines.append(f"名称：{req.name}")
    if req.description:
        user_lines.append(f"内容说明（剧情背景参考，若与用户当前版本的提示词冲突，以提示词为准）：{req.description}")
    if t == "storyboard":
        user_lines.append(f"台词（仅用于理解剧情，禁止写进画面）：{req.dialogue or '（无）'}")
        user_lines.append(f"该镜头时长：{req.duration} 秒")
    if req.old_prompt:
        user_lines.append(f"用户当前版本的提示词（最高优先级，其中用户写下的具体细节必须逐字保留）：{req.old_prompt}")
    if req.feedback:
        user_lines.append(f"用户的新要求（最高优先级，必须落实）：{req.feedback}")

    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": REGEN_SYSTEM_PROMPT},
                {"role": "user", "content": "\n".join(user_lines)},
            ],
            response_format={"type": "json_object"},
            max_tokens=1500,
            temperature=0.8,
        )
        data = deepseek_json(response)
        image_prompt = str(data.get("image_prompt") or "").strip()
        if not image_prompt:
            raise HTTPException(500, "重新生成失败：模型未返回提示词，请重试")
        log_prompt("重生成提示词", image_prompt, f"type={t}")
        return {
            "image_prompt": image_prompt,
            "video_prompt": str(data.get("video_prompt") or "").strip() if t == "storyboard" else "",
        }
    except ValueError as e:
        print(f"[regenerate-prompt] 输出解析失败：{e}")
        raise HTTPException(500, "DeepSeek 返回的不是合法 JSON，请重试")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, deepseek_friendly_error(e))

# ---------- 2.6 调度参考图提示词：俯视调度图 / 轨迹调度图（供分镜生成视频时给 AI 参考） ----------
class TopViewRequest(BaseModel):
    kind: str = "topview"  # topview=俯视调度图 / trajectory=轨迹调度图
    description: str = ""
    dialogue: str | None = None
    duration: int = 3
    video_prompt: str | None = None
    scene_name: str | None = None
    scene_description: str | None = None
    characters: List[Dict[str, Any]] | None = None
    style_bible: str | None = None
    old_prompt: str | None = None
    feedback: str | None = None
    base_plot: Dict[str, Any] | None = None  # 现有俯视调度图坐标（轨迹图以此为权威起点，不自由发挥）

TOPVIEW_SYSTEM_PROMPT = """你是动画导演的调度图纸绘制员。任务：为「俯视调度图」撰写文生图提示词。最终图像是一张【导演工作图纸】——从正上方垂直向下看的平面布置图（top-down floor plan），不是一幅场景画。

【image_prompt 撰写要求】单行输出，依次包含：
1. 图型声明（必须放最前、语气绝对）：纯正上方垂直俯视的平面调度图纸（top-down floor plan，正交投影、无透视、无景深），白底极简图纸风格，像建筑平面图/桌游沙盘顶视图；禁止画成立体透视场景，禁止动漫画面，禁止真实人物立绘。
2. 平面布局：按场景描述用简化几何形状（线框/浅色色块）画出地面边界与主要陈设的俯视投影（桌、床、门、窗、围栏、楼梯等），可加简短名称标注；不做细节刻画。
3. 人物棋子：每个出场角色画一个小型俯视人偶棋子图标（圆头+肩形剪影）并配角色名文字标签；图标位置即人物站位，用小箭头表示朝向；剧情中有移动的，用带箭头虚线画出运动路线。
4. 摄影机：用「机位」符号标出摄影机位置，并在符号处画一个扇形/梯形视锥表示拍摄朝向与视野范围；机位有移动的用箭头标注起止。
5. 文字约束：图上文字只允许角色名/陈设名/「机位」等简体中文短标签，禁止长句、字幕、水印。

【用户口述与修改要求】若提供「用户的新要求」，那是导演的明确调度指令（位置关系、朝向、机位或运动轨迹），必须逐条严格执行并完整体现在图中，与镜头描述冲突时以口述为准；旧提示词仅作基础参考。未提供口述时，按镜头描述自行编排合理调度。
【输出】严格输出 JSON：{"image_prompt": "..."}，不要 markdown，不要任何其他文字。"""

TRAJECTORY_SYSTEM_PROMPT = """你是动画导演的动作设计与运镜顾问。任务：为「轨迹调度图」撰写文生图提示词。最终图像是一张【导演工作图纸】——正上方垂直俯视的平面运动轨迹图（与俯视调度图同一视角、同一坐标系，可叠合使用），不是一幅场景画。

【image_prompt 撰写要求】单行输出，依次包含：
1. 图型声明（必须放最前、语气绝对）：纯正上方垂直俯视的运动轨迹图纸（top-down floor plan，正交投影、无透视、无景深），白底极简图纸风格，彩色轨迹线醒目；禁止画成立体透视场景，禁止动漫画面，禁止真实人物立绘。
2. 平面底图：用简化几何形状（线框/浅色色块）勾勒场景地面边界与关键陈设的俯视投影（桌、门、楼梯等）并标注名称，不做细节刻画。
3. 角色轨迹：每个出场角色用不同颜色，从起点到终点画一条连续的轨迹线（实线箭头），轨迹上用时间刻度小圆点标出关键节点（如 0 秒起点、中途转折、终点），节点旁标注动作短语（如"起跑""转身""急停"）；起点终点画人偶棋子图标+角色名标签。
4. 镜头轨迹：摄影机位置画「机位」符号+扇形视锥表示拍摄朝向，机位运动（推/拉/摇/移/跟）用黑色虚线箭头单独画出，标注方向与起止，与彩色角色轨迹区分。
5. 速度与节奏：在轨迹线上标注快慢（如"慢走""冲刺""急停"）。
6. 文字约束：图上文字只允许角色名/动作/秒数/「机位」等简体中文短标签，禁止长句、字幕、水印。

【用户口述与修改要求】若提供「用户的新要求」，那是导演的明确调度指令（位置关系、朝向、机位或运动轨迹），必须逐条严格执行并完整体现在图中，与镜头描述冲突时以口述为准；旧提示词仅作基础参考。未提供口述时，按镜头描述自行编排合理调度。
【输出】严格输出 JSON：{"image_prompt": "..."}，不要 markdown，不要任何其他文字。"""

def _fallback_diagram_prompt(req: TopViewRequest) -> str:
    chars = "、".join(str((c or {}).get("name") or "").strip() for c in (req.characters or []) if (c or {}).get("name"))
    parts = []
    if (req.kind or "").lower() == "trajectory":
        parts.append("纯正上方垂直俯视的运动轨迹图纸（top-down floor plan，正交投影、无透视、无景深），白底极简图纸风格，彩色轨迹线醒目；禁止立体透视场景，禁止动漫画面，禁止真实人物立绘")
        parts.append("平面底图：简化几何线框勾勒场景地面边界与关键陈设的俯视投影")
        parts.append(f"角色运动轨迹（每人一种颜色，起点到终点实线箭头，时间刻度圆点标注关键动作节点，起终点画人偶棋子图标+名字标签）：{chars or '按画面描述标注'}")
        parts.append("摄影机位置画「机位」符号+扇形视锥，机位运动用黑色虚线箭头单独画出并标注方向与起止")
        parts.append(f"本镜头运动内容：{req.video_prompt or req.description[:150]}")
    else:
        parts.append("纯正上方垂直俯视的平面调度图纸（top-down floor plan，正交投影、无透视、无景深），白底极简图纸风格，像建筑平面图；禁止立体透视场景，禁止动漫画面，禁止真实人物立绘")
        if req.scene_description:
            parts.append(f"平面布局（简化几何线框/浅色块画陈设俯视投影并标注名称）：{req.scene_description[:120]}")
        parts.append(f"本镜头调度：{req.description[:150]}")
        parts.append(f"出场角色（俯视人偶棋子图标+名字标签，标明站位与朝向，移动路线用虚线箭头）：{chars or '按画面描述标注'}")
        parts.append("用「机位」符号标出摄影机位置，并画扇形视锥表示拍摄朝向与视野")
    parts.append("图上文字仅限角色名/陈设名/动作/秒数/「机位」的简体中文短标签，禁止长句与水印")
    return "；".join(p for p in parts if p)

@app.post("/api/diagram-prompt")
async def diagram_prompt(req: TopViewRequest):
    """俯视调度图 / 轨迹调度图 的文生图提示词编排。DeepSeek 失败时用本地模板兜底，保证功能永远可用。"""
    kind = (req.kind or "topview").lower()
    system_prompt = TRAJECTORY_SYSTEM_PROMPT if kind == "trajectory" else TOPVIEW_SYSTEM_PROMPT
    user_lines = []
    # 注意：调度图是导演工作图纸，不套用 style_bible 成片画风（避免被带成透视场景画）
    if req.scene_name:
        user_lines.append(f"所在场景：{req.scene_name}")
    if req.scene_description:
        user_lines.append(f"场景描述：{req.scene_description}")
    user_lines.append(f"镜头画面描述：{req.description or '（无）'}")
    user_lines.append(f"台词（仅供理解剧情，不要成为图上文字）：{req.dialogue or '（无）'}")
    user_lines.append(f"镜头时长：{req.duration} 秒")
    if req.video_prompt:
        user_lines.append(f"视频运动描述：{req.video_prompt}")
    if req.characters:
        chars = "；".join(f"{str(c.get('name') or '')}：{str(c.get('description') or '')[:60]}" for c in req.characters if c.get("name"))
        if chars:
            user_lines.append(f"出场角色：{chars}")
    if req.old_prompt:
        user_lines.append(f"旧提示词（在其基础上重写）：{req.old_prompt}")
    if req.feedback:
        user_lines.append(f"用户的新要求（最高优先级）：{req.feedback}")

    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "\n".join(user_lines)},
            ],
            response_format={"type": "json_object"},
            max_tokens=1000,
            temperature=0.6,
        )
        data = deepseek_json(response)
        prompt = str(data.get("image_prompt") or "").strip()
        if prompt:
            log_prompt(f"调度图提示词({kind})", prompt)
            return {"image_prompt": prompt, "source": "ai"}
    except Exception as e:
        print(f"[diagram-prompt] DeepSeek 生成失败（{kind}），用模板兜底: {type(e).__name__}: {e}")
    fb = _fallback_diagram_prompt(req)
    log_prompt(f"调度图提示词({kind}, 模板兜底)", fb)
    return {"image_prompt": fb, "source": "template"}

# ---------- 2.62 调度图纸（新方案）：DeepSeek 解析口述为坐标数据，PIL 程序化绘制平面图纸 ----------
# AI 生图画"俯视图"总会跑偏成透视场景照片，故改为：AI 只负责把口述转成结构化坐标，图纸由代码精确绘制。

DIAGRAM_PLOT_PROMPT = """你是调度图纸数据员。把导演的调度口述与镜头描述解析为俯视平面图纸的结构化坐标数据（图纸由程序绘制，你只产出数据）。
【坐标系】x：0=图纸左，1=图纸右；y：0=图纸上方（空间远端），1=图纸下方（空间近端）。角度：0=朝图纸上方，90=朝右，180=朝下方，270=朝左，顺时针，单位度。
【输出格式】严格只输出 JSON，不要 markdown，不要其他文字：
{
  "props": [{"label":"名称","shape":"rect","x":0.5,"y":0.5,"w":0.2,"h":0.15}],
  "characters": [{"name":"角色名","x":0.3,"y":0.6,"facing":0,"path":[[0.3,0.6],[0.3,0.4]],"stops":[{"t":"0秒","action":"起跑"},{"t":"3秒","action":"急停"}]}],
  "camera": {"x":0.5,"y":0.95,"facing":0,"fov":55,"path":[],"movement":"固定机位"},
  "note": "一句话备注，可为空字符串"
}
【规则】
1. 导演口述是最高优先级，逐条落实为坐标；口述中的「画左/画右/画上方/画下方」直接对应图纸左/右/上/下。
2. props：按场景描述列出主要陈设的俯视投影（水箱、桌、床、门等），x,y 为中心点、w,h 为宽高（均为 0-1）。
3. characters：只包含出场角色；口述未提及位置的角色按剧情合理推断放置，并在 note 中注明。有移动的给 path（按时间顺序 2-6 个拐点）与 stops（与拐点对应）。
4. camera：机位位置与拍摄朝向；movement 为「固定机位」时 path 为空数组；fov 缺省 55。
5. 所有坐标数值必须在 0-1 范围内。
6. 若用户提供「现有俯视调度图坐标」（权威起点数据）：它是同一空间同一时刻的既成事实，所有角色的起点 x,y 与朝向必须照抄该数据中同名角色的值，严禁重新安排站位或凭剧情重新布局；轨迹 path 必须从该起点出发，只解析口述中提到的移动/朝向变化，未提及移动的角色停在原位（path 空数组、朝向照抄）；陈设 props 与机位起点同样照抄该数据，仅当口述明确说机位移动时才给 camera.path。"""

def _fallback_plot(req: TopViewRequest) -> dict:
    """DeepSeek 解析失败时的默认布局（明确告知用户重试）。"""
    chars = [str((c or {}).get("name") or "").strip() for c in (req.characters or []) if (c or {}).get("name")]
    chars = [c for c in chars if c]
    n = max(len(chars), 1)
    characters = []
    for i, name in enumerate(chars):
        row = i // 5
        col = i % 5
        cnt = min(n - row * 5, 5)
        characters.append({
            "name": name, "x": (col + 1) / (cnt + 1), "y": 0.45 + row * 0.25,
            "facing": 180 if row == 0 else 0, "path": [], "stops": [],
        })
    return {
        "props": [], "characters": characters,
        "camera": {"x": 0.5, "y": 0.95, "facing": 0, "fov": 55, "path": [], "movement": "固定机位"},
        "note": "AI 解析口述失败，已用默认布局——请检查口述后重新生成",
    }

def _clamp01(v) -> float:
    try:
        return min(1.0, max(0.0, float(v)))
    except (TypeError, ValueError):
        return 0.5


def _merge_base_plot(plot: dict, base: dict) -> dict:
    """轨迹图以现有俯视调度图为起点：角色起点/朝向/陈设/机位起点强制对齐 base 坐标。
    AI 解析结果只贡献「运动轨迹与朝向变化」，防止其自由发挥重新布局。"""
    def _near(a, b, tol=0.09) -> bool:
        try:
            return abs(float(a) - float(b)) <= tol
        except (TypeError, ValueError):
            return False

    bchars = {}
    for c in (base.get("characters") or []):
        if isinstance(c, dict) and str(c.get("name") or "").strip():
            bchars[str(c.get("name")).strip()] = c

    merged, seen = [], set()
    for c in (plot.get("characters") or []):
        if not isinstance(c, dict) or not str(c.get("name") or "").strip():
            continue
        name = str(c.get("name")).strip()
        b = bchars.get(name)
        if not b:
            merged.append(c)
            continue
        seen.add(name)
        path = [list(pt)[:2] for pt in (c.get("path") or [])
                if isinstance(pt, (list, tuple)) and len(pt) >= 2]
        if path:
            # 轨迹必须从 base 起点出发：首个拐点偏离起点则补插起点
            if not (_near(path[0][0], b.get("x")) and _near(path[0][1], b.get("y"))):
                path = [[b.get("x"), b.get("y")]] + path
            # 有移动：起点照抄 base，终点朝向用 AI 解析的（转身等朝向变化）
            merged.append({**c, "x": b.get("x"), "y": b.get("y"), "path": path})
        else:
            # 口述未提及移动：停在原位、朝向照抄 base
            merged.append({**c, "x": b.get("x"), "y": b.get("y"),
                           "facing": b.get("facing", c.get("facing")), "path": [], "stops": []})
    # base 里有、口述没提的角色：原位保留（两图人物一致，可叠合对照）
    for name, b in bchars.items():
        if name not in seen:
            merged.append({**b, "path": [], "stops": []})
    plot["characters"] = merged

    # 陈设照抄 base：同一场景，两图坐标系一致
    if base.get("props"):
        plot["props"] = base.get("props")

    # 机位起点照抄 base；仅当 AI 解析出机位移动路径时才允许动
    bc = base.get("camera") if isinstance(base.get("camera"), dict) else {}
    if bc:
        pc = plot.get("camera") if isinstance(plot.get("camera"), dict) else {}
        pc["x"], pc["y"] = bc.get("x"), bc.get("y")
        cam_path = [list(pt)[:2] for pt in (pc.get("path") or [])
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2]
        if cam_path:
            if not (_near(cam_path[0][0], bc.get("x")) and _near(cam_path[0][1], bc.get("y"))):
                cam_path = [[bc.get("x"), bc.get("y")]] + cam_path
            pc["path"] = cam_path
        else:
            pc["facing"] = bc.get("facing", pc.get("facing"))
            if not str(pc.get("movement") or "").strip() or "固定" in str(pc.get("movement")):
                pc["movement"] = bc.get("movement") or "固定机位"
        plot["camera"] = pc
    return plot

def _normalize_plot(plot: dict) -> dict:
    props = []
    for p in (plot.get("props") or [])[:12]:
        if not isinstance(p, dict):
            continue
        props.append({
            "label": str(p.get("label") or "陈设")[:8],
            "shape": "rect",
            "x": _clamp01(p.get("x")), "y": _clamp01(p.get("y")),
            "w": min(1.0, max(0.02, _clamp01(p.get("w")) or 0.15)),
            "h": min(1.0, max(0.02, _clamp01(p.get("h")) or 0.1)),
        })
    characters = []
    for c in (plot.get("characters") or [])[:10]:
        if not isinstance(c, dict) or not str(c.get("name") or "").strip():
            continue
        path = []
        for pt in (c.get("path") or [])[:8]:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                path.append([_clamp01(pt[0]), _clamp01(pt[1])])
        stops = []
        for s in (c.get("stops") or [])[:8]:
            if isinstance(s, dict) and (str(s.get("action") or "").strip() or str(s.get("t") or "").strip()):
                stops.append({"t": str(s.get("t") or "")[:6], "action": str(s.get("action") or "")[:8]})
        facing = float(c.get("facing") or 0)
        characters.append({
            "name": str(c.get("name"))[:6],
            "x": _clamp01(c.get("x")), "y": _clamp01(c.get("y")),
            "facing": facing % 360,
            "path": path, "stops": stops,
        })
    cam = plot.get("camera") if isinstance(plot.get("camera"), dict) else {}
    cam_path = []
    for pt in (cam.get("path") or [])[:8]:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            cam_path.append([_clamp01(pt[0]), _clamp01(pt[1])])
    camera = {
        "x": _clamp01(cam.get("x", 0.5)), "y": _clamp01(cam.get("y", 0.95)),
        "facing": float(cam.get("facing") or 0) % 360,
        "fov": min(120.0, max(15.0, float(cam.get("fov") or 55))),
        "path": cam_path,
        "movement": str(cam.get("movement") or "固定机位")[:10],
    }
    return {"props": props, "characters": characters, "camera": camera,
            "note": str(plot.get("note") or "")[:80]}

# 角色图标配色池（醒目、互相区分）
_DIAGRAM_PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
                    "#008080", "#f032e6", "#9a6324", "#000075", "#800000"]

def _frame_side_text(cross: float, dist: float, fov: float) -> str:
    """把机位相对的左右偏角换成生图模型能落地的画面落位描述。
    cross/dist ＝ 人物相对视轴的带符号正弦（>0 视轴右侧）。
    近轴（几乎正对镜头）时必须说'画面中部'——若硬说'画面左/右'，
    模型会把几度的微小偏差放大成明显靠边，构图直接跑偏。"""
    import math
    if dist <= 1e-6:
        return "画面中部"
    lat_deg = math.degrees(math.asin(max(-1.0, min(1.0, cross / dist))))  # >0=视轴右侧
    if abs(lat_deg) < 8:
        return "画面中部（几乎正对镜头）"
    edge = "、靠近画面边缘" if abs(lat_deg) > max(fov / 2, 30) else ""
    return ("画面右侧" if lat_deg > 0 else "画面左侧") + edge


def _camera_view_summary(plot: dict) -> str:
    """按坐标几何计算机位视野：谁在镜头前方（入画，近→远）、谁在机位旁/镜头后方（绝不入画）。
    这是权威事实，直接注入分镜生图约束，杜绝"从主角位置拍过去"的反向理解。"""
    import math
    cam = plot.get("camera") or {}
    cx, cy = float(cam.get("x", 0.5)), float(cam.get("y", 0.95))
    facing = float(cam.get("facing") or 0)
    fov = float(cam.get("fov") or 55)
    rad = math.radians(facing)
    fx, fy = math.sin(rad), -math.cos(rad)
    dir_desc = {0: "图纸上方", 45: "图纸右上方", 90: "图纸右方", 135: "图纸右下方",
                180: "图纸下方", 225: "图纸左下方", 270: "图纸左方", 315: "图纸左上方"}
    dir_name = dir_desc[min(((abs((facing % 360) - k), k) for k in dir_desc))[1]]

    visible, hidden = [], []
    for c in plot.get("characters") or []:
        name = str(c.get("name") or "").strip()
        if not name:
            continue
        vx, vy = float(c.get("x", 0)) - cx, float(c.get("y", 0)) - cy
        dist = math.hypot(vx, vy)
        if dist < 1e-6:
            hidden.append((name, 0.0, None, "就在机位上（视角人物，镜头从他眼中拍出）"))
            continue
        dot = (vx * fx + vy * fy) / dist
        angle = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
        # cross>0 → 人物在机位视线右侧（即成片画面右侧）
        side = _frame_side_text(fx * vy - fy * vx, dist, fov)
        if dist <= 0.15:
            hidden.append((name, dist, None, "紧贴机位（就在镜头旁边，不入画）"))
        elif dot <= 0:
            hidden.append((name, dist, None, "在镜头后方（机位身后，不入画）"))
        elif angle > fov / 2 + 25:
            hidden.append((name, dist, None, "在机位视野范围之外（不入画）"))
        else:
            facing_desc = _facing_relative_to_camera(float(c.get("facing") or 0), vx, vy, dist, rad)
            visible.append((name, dist, angle, side, facing_desc))
    visible.sort(key=lambda t: t[1])

    lines = [f"镜头架在「机位」符号处、朝{dir_name}方向拍摄（视锥约{int(fov)}°）——画面就是从机位望出去的景象。"]
    if visible:
        n = max(len(visible) - 1, 1)
        seg = []
        for i, (name, dist, angle, side, facing_desc) in enumerate(visible):
            depth = "离镜头最近" if i == 0 else ("最远" if i == n else "中距")
            vert = "位于画面下缘" if i == 0 else ("靠近画面上缘地平线" if i == n else "位于画面纵深中部")
            edge = "，靠近视野边缘可能被裁切" if angle > fov / 2 + 10 else ""
            seg.append(f"{name}（{vert}、{depth}、{side}、{facing_desc}{edge}）")
        lines.append("机位前方视野内、会出现在画面中（从近到远）：" + "、".join(seg) + "。")
    else:
        lines.append("机位前方视野内没有人物（画面以环境空镜为主）。")
    if hidden:
        lines.append("以下人物站在机位上/机位旁/镜头后方或视野之外，绝不能出现在画面中：" +
                     "；".join(f"{n}（{why}）" for n, _, _, why in hidden) + "。")
    lines.append("此名单由程序按调度图纸坐标计算，是权威事实：机位的位置与朝向、谁入画谁不入画、每人的画面左右与身体朝向都以本事实为准；"
                 "导演口述文字若与图纸「机位」符号不符（例如导演手动移过机位后口述仍写旧位置），口述中的机位描述作废，按本事实执行。")
    return "".join(lines)


def _visible_character_names(plot: dict) -> list[str] | None:
    """按机位几何计算本镜头"入画人物名单"（与 _camera_view_summary 同一套判定）。
    无有效 plot 数据时返回 None（表示不过滤）。视角人物（站在机位上）不入画。"""
    import math
    if not isinstance(plot, dict):
        return None
    chars = [c for c in (plot.get("characters") or []) if str(c.get("name") or "").strip()]
    if not chars:
        return None
    cam = plot.get("camera") or {}
    cx, cy = float(cam.get("x", 0.5)), float(cam.get("y", 0.95))
    rad = math.radians(float(cam.get("facing") or 0))
    fx, fy = math.sin(rad), -math.cos(rad)
    fov = float(cam.get("fov") or 55)
    visible = []
    for c in chars:
        name = str(c.get("name")).strip()
        vx, vy = float(c.get("x", 0)) - cx, float(c.get("y", 0)) - cy
        dist = math.hypot(vx, vy)
        if dist < 1e-6:
            continue  # 视角人物：镜头从他眼中拍出，本人不出现在画面
        dot = (vx * fx + vy * fy) / dist
        angle = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
        if dist > 0.15 and dot > 0 and angle <= fov / 2 + 25:
            visible.append(name)
    return visible


def _facing_relative_to_camera(pf: float, vx: float, vy: float, dist: float, cam_rad: float) -> str:
    """把人物在图纸上的朝向（0=图纸上方、顺时针）换算成生图模型听得懂的镜头相对朝向：
    面向镜头 / 背对镜头 / 侧对镜头（面朝画面左或右）。"""
    import math
    if dist <= 1e-6:
        return "视角人物（镜头从他眼中拍出）"
    prad = math.radians(pf % 360)
    pfx, pfy = math.sin(prad), -math.cos(prad)  # 人物面朝方向（图纸坐标）
    to_cam = (-vx / dist, -vy / dist)           # 人物→机位方向（看向镜头的方向）
    dot = pfx * to_cam[0] + pfy * to_cam[1]
    if dot >= 0.5:
        return "面向镜头"
    if dot <= -0.5:
        return "背对镜头"
    # 侧对：机位视线右侧在图纸上的方向向量（rad=0 朝上拍时=图纸右方 (1,0)）
    rx, ry = math.cos(cam_rad), math.sin(cam_rad)
    side_dot = pfx * rx + pfy * ry
    return "侧对镜头、面朝画面右" if side_dot > 0 else "侧对镜头、面朝画面左"


def format_plot_to_prompt(plot: dict) -> str:
    """把调度图精准坐标数据翻译成生图模型听得懂的自然语言（注入 prompt 尾部）。
    生图模型读不懂白底俯视图上的圆点与扇形，但读得懂"谁离镜头最近、在画面哪一侧"。"""
    import math
    if not isinstance(plot, dict):
        return ""
    cam = plot.get("camera") or {}
    chars = [c for c in (plot.get("characters") or []) if str(c.get("name") or "").strip()]
    if not chars and not cam:
        return ""
    cx, cy = float(cam.get("x", 0.5)), float(cam.get("y", 0.95))
    facing = float(cam.get("facing") or 0) % 360
    fov = float(cam.get("fov") or 55)
    rad = math.radians(facing)
    fx, fy = math.sin(rad), -math.cos(rad)
    dir_desc = {0: "图纸上方", 45: "图纸右上方", 90: "图纸右方", 135: "图纸右下方",
                180: "图纸下方", 225: "图纸左下方", 270: "图纸左方", 315: "图纸左上方"}
    dir_name = dir_desc[min(((abs(facing - k), k) for k in dir_desc))[1]]

    lines = [f"【强制机位】镜头架在「机位」符号处，朝{dir_name}方向拍摄，视锥约{int(fov)}°。画面＝从机位望出去的景象。"]

    # 人物站位：按与机位的几何关系逐人描述（前后/左右/远近排序），生图模型听不懂裸坐标，只听得懂相对关系
    measured = []
    for c in chars:
        name = str(c.get("name")).strip()
        px, py = float(c.get("x", 0)), float(c.get("y", 0))
        vx, vy = px - cx, py - cy
        dist = math.hypot(vx, vy)
        if dist < 1e-6:
            measured.append((name, 0.0, None, None, "就站在机位上（视角人物，镜头从他眼中拍出）"))
            continue
        dot = (vx * fx + vy * fy) / dist
        angle = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
        raw_side = "画面右" if (fx * vy - fy * vx) > 0 else "画面左"
        side = _frame_side_text(fx * vy - fy * vx, dist, fov)
        pf = float(c.get("facing") or 0) % 360
        pdir = dir_desc[min(((abs(pf - k), k) for k in dir_desc))[1]].replace("图纸", "面朝图纸")
        orient = _facing_relative_to_camera(pf, vx, vy, dist, rad)
        if dist <= 0.15:
            measured.append((name, dist, None, None, f"紧贴机位{raw_side}侧（就在镜头旁边，不入画），{orient}"))
        elif dot <= 0:
            measured.append((name, dist, None, None, f"在机位身后（镜头后方，不入画），{orient}"))
        elif angle > fov / 2 + 25:
            measured.append((name, dist, None, None, f"在机位视野之外（不入画），{orient}"))
        else:
            measured.append((name, dist, angle, side, f"在机位前方{side}、{orient}（图纸方位：{pdir}）"))

    visible = sorted([m for m in measured if "前方" in m[4]], key=lambda t: t[1])
    hidden = [m for m in measured if "前方" not in m[4]]
    n = max(len(visible) - 1, 1)
    seg = []
    for i, (name, dist, angle, side, desc) in enumerate(visible):
        depth = "离镜头最近" if i == 0 else ("离镜头最远" if i == n else "中距")
        vert = "位于画面下缘" if i == 0 else ("靠近画面上缘地平线" if i == n else "位于画面纵深中部")
        edge = "、靠近视野边缘可能被画面裁切" if angle and angle > fov / 2 + 10 else ""
        seg.append(f"{name}（{vert}、{depth}、{desc}{edge}）")
    if seg:
        lines.append("【人物站位·入画名单（从近到远）】（本名单中的「画面左/右」＝成片图像里观众看到的左/右，"
                     "已按机位朝向换算，与图纸方位无关）" + "；".join(seg) + "。")
    else:
        lines.append("【人物站位】机位前方视野内没有人物，画面以环境空镜为主。")
    if hidden:
        lines.append("【人物站位·绝不入画名单】" + "；".join(f"{m[0]}（{m[4]}）" for m in hidden) + "。")
    lines.append("【透视规则】离机位越近的人画面里越大（可为前景，可能只露腿部/肩膀/背影），越远越小、靠近地平线；"
                 "同一地面人物身高一致，纵深相近者大小相近。"
                 "每个人物的身体朝向以上述描述为准（面向镜头＝正脸朝观众，背对镜头＝只见背影，侧对镜头＝按写明的画面左右侧脸），"
                 "文字提示词中的朝向描述（如'面朝XX方向''看画面的对侧''望向XX'）若与本名单冲突，一律作废、按本名单执行。"
                 "镜头就是对准入画名单里的人拍的：离镜头最近的主体应按景别呈现在镜头前方（构图围绕该主体），"
                 "写'面向镜头'的主体必须正脸朝向镜头，严禁把他挪到画面边缘、画成侧脸、或把画面取成顺着他视线望出去的方向。"
                 "上面两个名单由程序按坐标计算，是权威事实："
                 "绝不入画名单里的人禁止出现在画面中，即使文字提示词提到了他们。"
                 "每个角色在画面中只出现一次、只有一个身体：走位描述的是同一角色的位移，严禁画成分身、复制人或两个同名角色；"
                 "俯视调度图纸/轨迹图类参考图仅用于确定站位与走位，严禁把图纸、人偶棋子图标、扇形视锥画进画面。")

    # 运动路线（轨迹图）：拐点翻译成镜头相对方向（往画面哪边走、靠近/远离镜头），不甩裸坐标
    cam_facing = float(cam.get("facing") or 0)

    def _dir8(dx: float, dy: float) -> str:
        """位移向量 → 镜头相对运动方向描述（以机位视线为正前方）。"""
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return ""
        ang = (math.degrees(math.atan2(dx, -dy)) - cam_facing) % 360
        sectors = [
            (22.5, "向画面纵深走去（远离镜头）"), (67.5, "向画面右后方走（远离镜头）"),
            (112.5, "向画面右走"), (157.5, "向画面右前方走来（靠近镜头）"),
            (202.5, "朝镜头走来（靠近镜头）"), (247.5, "向画面左前方走来（靠近镜头）"),
            (292.5, "向画面左走"), (337.5, "向画面左后方走（远离镜头）"),
        ]
        for lim, txt in sectors:
            if ang < lim:
                return txt
        return "向画面纵深走去（远离镜头）"

    def _depth_trend(pts) -> str:
        """起点→终点相对机位的纵深变化。"""
        if len(pts) < 2:
            return ""
        d0 = math.hypot(pts[0][0] - cx, pts[0][1] - cy)
        d1 = math.hypot(pts[-1][0] - cx, pts[-1][1] - cy)
        if d1 < d0 - 0.05:
            return "整体逐渐靠近镜头（人物在画面里逐渐变大）"
        if d1 > d0 + 0.05:
            return "整体逐渐远离镜头（人物在画面里逐渐变小）"
        return "纵深基本持平"

    moves = []
    for c in chars:
        path = c.get("path") or []
        if not (path and str(c.get("name")).strip()):
            continue
        pts = [[float(c.get("x") or 0), float(c.get("y") or 0)]] + [list(p)[:2] for p in path]
        segs = []
        for a, b in zip(pts, pts[1:]):
            d = _dir8(b[0] - a[0], b[1] - a[1])
            if d and (not segs or segs[-1] != d):
                segs.append(d)
        desc = str(c.get("name")).strip() + "从起点出发移动：" + "→".join(segs)
        trend = _depth_trend(pts)
        if trend:
            desc += f"，{trend}"
        stops = [s for s in (c.get("stops") or []) if isinstance(s, dict) and (str(s.get("action") or "").strip() or str(s.get("t") or "").strip())]
        if stops:
            desc += "；时间节点：" + "→".join(f"{str(s.get('t') or '').strip()}{str(s.get('action') or '').strip()}".strip() for s in stops)
        moves.append(desc)
    cam_path = cam.get("path") or []
    if cam_path:
        cam_pts = [[float(cam.get("x") or 0), float(cam.get("y") or 0)]] + [list(p)[:2] for p in cam_path]
        d = _dir8(cam_pts[-1][0] - cam_pts[0][0], cam_pts[-1][1] - cam_pts[0][1])
        moves.append(f"机位{d or '移动'}（{(cam.get('movement') or '移动机位')}）")
    if moves:
        lines.append("【运动路线】" + "；".join(moves) + "。运动要平滑连续，走位方向与快慢以上述路线为准。")
    return "\n".join(lines) + "\n"


def _load_cn_font(size: int):
    from PIL import ImageFont
    for name in ("msyh.ttc", "simhei.ttf", "simsun.ttc", "arial.ttf"):
        try:
            return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)
        except Exception:
            continue
    return ImageFont.load_default()

def _hex_rgb(c: str):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))

def render_diagram_png(plot: dict, kind: str) -> bytes:
    """把调度坐标数据绘制成平面图纸 PNG。坐标系：x 右为正、y 下为正（0-1 归一化）。"""
    from PIL import Image, ImageDraw
    import math

    W, H = 1400, 1460
    M = 40                      # 图纸外边距
    TITLE_H = 76
    LEGEND_H = 110 if plot.get("characters") else 46
    FX0, FY0 = M, TITLE_H       # 场景平面区域
    FW, FH = W - M * 2, H - TITLE_H - LEGEND_H - M - 30
    img = Image.new("RGB", (W, H), "#ffffff")
    d = ImageDraw.Draw(img)
    f_title = _load_cn_font(34)
    f_label = _load_cn_font(24)
    f_small = _load_cn_font(19)
    f_tag = _load_cn_font(22)

    is_traj = (kind or "").lower() == "trajectory"
    d.text((M, 18), ("轨迹调度图 · 运动轨迹与机位移动（正上方俯视平面图）" if is_traj
                     else "俯视调度图 · 人物位置与机位（正上方俯视平面图）"),
           fill="#1f2937", font=f_title)
    d.text((W - M, 30), "上=空间远端  下=空间近端", fill="#9ca3af", font=f_small, anchor="ra")

    # 场景地面边框 + 网格
    d.rounded_rectangle([FX0, FY0, FX0 + FW, FY0 + FH], radius=18, outline="#374151", width=5)
    for i in range(1, 10):
        gx = FX0 + FW * i / 10
        gy = FY0 + FH * i / 10
        d.line([gx, FY0 + 4, gx, FY0 + FH - 4], fill="#f1f2f4", width=2)
        d.line([FX0 + 4, gy, FX0 + FW - 4, gy], fill="#f1f2f4", width=2)

    def P(x, y):
        return (FX0 + FW * x, FY0 + FH * y)

    def dashed_line(p1, p2, color, width=5, dash=16, gap=10):
        x1, y1 = p1; x2, y2 = p2
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist < 1:
            return
        ux, uy = (x2 - x1) / dist, (y2 - y1) / dist
        t = 0.0
        while t < dist:
            t2 = min(t + dash, dist)
            d.line([x1 + ux * t, y1 + uy * t, x1 + ux * t2, y1 + uy * t2], fill=color, width=width)
            t = t2 + gap

    def arrow_head(tip, angle_deg, color, size=20):
        ax, ay = tip
        rad = math.radians(angle_deg)
        for spread in (150, 210):
            wx = ax + size * math.cos(rad + math.radians(spread))
            wy = ay + size * math.sin(rad + math.radians(spread))
            d.line([ax, ay, wx, wy], fill=color, width=width_of(color))

    def width_of(_c):
        return 5

    def facing_xy(x, y, facing, length):
        rad = math.radians(facing)
        return (x + length * math.sin(rad), y - length * math.cos(rad))

    # 陈设
    for p in plot.get("props") or []:
        px, py = P(p["x"], p["y"])
        pw, ph = FW * p["w"], FH * p["h"]
        d.rectangle([px - pw / 2, py - ph / 2, px + pw / 2, py + ph / 2],
                    fill="#eef1f5", outline="#8a93a3", width=3)
        d.text((px, py - ph / 2 - 16), p["label"], fill="#6b7280", font=f_tag, anchor="mm")

    # 角色轨迹（先画线，再画人偶，线在人偶下方）
    chars = plot.get("characters") or []
    color_of = {c["name"]: _DIAGRAM_PALETTE[i % len(_DIAGRAM_PALETTE)] for i, c in enumerate(chars)}
    for c in chars:
        col = color_of[c["name"]]
        path_pts = [P(*pt) for pt in (c.get("path") or [])]
        if path_pts:
            full = [P(c["x"], c["y"])] + path_pts
            for i in range(len(full) - 1):
                dashed_line(full[i], full[i + 1], col, width=6)
            arrow_head(full[-1], math.degrees(math.atan2(full[-1][1] - full[-2][1], full[-1][0] - full[-2][0])), col)
            # 时间节点
            stops = c.get("stops") or []
            for i, pt in enumerate(full):
                d.ellipse([pt[0] - 9, pt[1] - 9, pt[0] + 9, pt[1] + 9], fill=col, outline="#ffffff", width=3)
                label = ""
                if i < len(stops):
                    label = f"{stops[i].get('t', '')} {stops[i].get('action', '')}".strip()
                if label:
                    d.text((pt[0] + 14, pt[1] - 14), label, fill=col, font=f_small, anchor="lm")

    # 机位轨迹（黑色虚线，画在人偶下方）
    cam = plot.get("camera") or {}
    cam_path_pts = [P(*pt) for pt in (cam.get("path") or [])]
    cam_pt = P(cam.get("x", 0.5), cam.get("y", 0.95))
    cam_full = [cam_pt] + cam_path_pts
    for i in range(len(cam_full) - 1):
        dashed_line(cam_full[i], cam_full[i + 1], "#111827", width=5, dash=12, gap=8)
        arrow_head(cam_full[i + 1], math.degrees(math.atan2(cam_full[i + 1][1] - cam_full[i][1], cam_full[i + 1][0] - cam_full[i][0])), "#111827", size=16)

    # 机位视锥（半透明感浅黄色扇形）
    fov = float(cam.get("fov") or 55)
    facing = float(cam.get("facing") or 0)
    R = min(FW, FH) * 0.38
    # PIL 角度以 3 点钟方向为 0、顺时针；本坐标系 0=朝上 → pil = facing - 90
    start = facing - fov / 2 - 90
    end = facing + fov / 2 - 90
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.pieslice([cam_pt[0] - R, cam_pt[1] - R, cam_pt[0] + R, cam_pt[1] + R],
                start, end, fill=(250, 214, 90, 88))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    d = ImageDraw.Draw(img)

    # 人物人偶棋子（起点位置；有轨迹的由轨迹线+箭头表达终点）
    char_pts = [(c, P(c["x"], c["y"])) for c in chars]
    for c, pt in char_pts:
        col = color_of[c["name"]]
        r = 17
        d.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill=col, outline="#ffffff", width=4)
        # 朝向箭头
        tip = facing_xy(pt[0], pt[1], c.get("facing", 0), 40)
        d.line([pt[0], pt[1], tip[0], tip[1]], fill=col, width=6)
        arrow_head(tip, c.get("facing", 0), col)
        # 名字标签
        label = c["name"]
        tw = d.textlength(label, font=f_label)
        lx, ly = pt[0] - tw / 2 - 10, pt[1] + r + 8
        d.rounded_rectangle([lx, ly, lx + tw + 20, ly + 34], radius=8, fill="#ffffff", outline=col, width=3)
        d.text((lx + 10, ly + 4), label, fill="#111827", font=f_label)

    # 机位符号（相机机身图形 + 文字）
    d.ellipse([cam_pt[0] - 16, cam_pt[1] - 16, cam_pt[0] + 16, cam_pt[1] + 16], fill="#111827", outline="#ffffff", width=3)
    d.rectangle([cam_pt[0] - 9, cam_pt[1] - 5, cam_pt[0] + 9, cam_pt[1] + 6], fill="#ffffff")
    d.ellipse([cam_pt[0] - 4, cam_pt[1] - 2, cam_pt[0] + 4, cam_pt[1] + 4], fill="#111827")
    d.text((cam_pt[0], cam_pt[1] + 24), "机位", fill="#111827", font=f_label, anchor="ma")

    # 图例
    ly0 = FY0 + FH + 18
    x = M
    d.text((x, ly0), "图例：", fill="#374151", font=f_tag)
    x += 92
    for c in chars:
        col = color_of[c["name"]]
        d.ellipse([x, ly0 + 4, x + 22, ly0 + 26], fill=col, outline="#ffffff", width=2)
        d.text((x + 30, ly0 + 3), c["name"], fill="#374151", font=f_tag)
        x += 30 + d.textlength(c["name"], font=f_tag) + 26
    d.ellipse([x, ly0 + 4, x + 22, ly0 + 26], fill="#111827")
    d.text((x + 30, ly0 + 3), "机位（扇形=视野）", fill="#374151", font=f_tag)
    x += 30 + d.textlength("机位（扇形=视野）", font=f_tag) + 26
    if is_traj:
        d.line([x, ly0 + 15, x + 44, ly0 + 15], fill="#111827", width=4)
        d.text((x + 52, ly0 + 3), "机位移动", fill="#374151", font=f_tag)

    note = str(plot.get("note") or "").strip()
    if note:
        d.text((M, H - 40), f"备注：{note}", fill="#6b7280", font=f_small)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

@app.post("/api/render-diagram")
async def render_diagram(req: TopViewRequest):
    """调度图纸：DeepSeek 解析口述 → 坐标数据 → PIL 程序化绘制。解析失败用默认布局兜底并提示重试。"""
    kind = (req.kind or "topview").lower()
    user_lines = []
    if req.scene_name:
        user_lines.append(f"所在场景：{req.scene_name}")
    if req.scene_description:
        user_lines.append(f"场景描述：{req.scene_description}")
    user_lines.append(f"镜头画面描述：{req.description or '（无）'}")
    user_lines.append(f"台词（仅供理解剧情）：{req.dialogue or '（无）'}")
    user_lines.append(f"镜头时长：{req.duration} 秒")
    if req.video_prompt:
        user_lines.append(f"视频运动描述：{req.video_prompt}")
    if req.characters:
        chars = "；".join(str(c.get("name") or "") for c in req.characters if c.get("name"))
        if chars:
            user_lines.append(f"出场角色：{chars}")
    if req.feedback:
        user_lines.append(f"导演口述（最高优先级，逐条落实为坐标）：{req.feedback}")
    base = req.base_plot if isinstance(req.base_plot, dict) else None
    if base:
        blines = []
        for c in (base.get("characters") or []):
            if isinstance(c, dict) and str(c.get("name") or "").strip():
                blines.append(f"- {c.get('name')}：起点({c.get('x')},{c.get('y')})，朝向{c.get('facing')}度")
        bc = base.get("camera") if isinstance(base.get("camera"), dict) else {}
        if bc:
            blines.append(f"- 机位：({bc.get('x')},{bc.get('y')})，朝向{bc.get('facing')}度，movement={bc.get('movement')}")
        if blines:
            user_lines.append("现有俯视调度图坐标（权威起点，角色起点/朝向/机位起点必须照抄，轨迹从此出发，禁止重新布局）：\n" + "\n".join(blines))

    plot, source = None, "ai"
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": DIAGRAM_PLOT_PROMPT},
                {"role": "user", "content": "\n".join(user_lines)},
            ],
            response_format={"type": "json_object"},
            max_tokens=1200,
            temperature=0.2,
        )
        data = deepseek_json(response)
        if isinstance(data.get("characters"), list) and data["characters"]:
            plot = data
    except Exception as e:
        print(f"[render-diagram] DeepSeek 解析失败（{kind}），用默认布局: {type(e).__name__}: {e}")
    if not plot:
        plot = _fallback_plot(req)
        source = "template"
    if base:
        # 程序层强制对齐：即使 AI 没照抄起点，也以 base 坐标为准合并
        plot = _merge_base_plot(plot, base)
    plot = _normalize_plot(plot)
    png = render_diagram_png(plot, kind)
    os.makedirs("static/uploads", exist_ok=True)
    fname = f"diagram_{kind}_{uuid.uuid4().hex[:10]}.png"
    with open(os.path.join("static", "uploads", fname), "wb") as f:
        f.write(png)
    log_prompt(f"调度图纸绘制({kind})", json.dumps(plot, ensure_ascii=False)[:500])
    cam_summary = _camera_view_summary(plot)
    log_prompt(f"机位视野计算({kind})", cam_summary)
    return {"image_url": f"/static/uploads/{fname}", "plot": plot, "cam_summary": cam_summary, "source": source}

# ---------- 2.62 手动移机位：导演在图上标注/拖动机位后，把 plot.camera 真正移过去并重绘图纸（消除双机位） ----------
class MoveCameraRequest(BaseModel):
    plot: dict  # /api/render-diagram 返回的坐标数据
    x: float  # 新机位 x（0-1，图纸坐标）
    y: float  # 新机位 y（0-1，图纸坐标）
    facing: float | None = None  # 新机位朝向（度，0=图纸上方、90=右方、180=下方、270=左方）；None=保持原朝向

@app.post("/api/move-camera")
async def move_camera(req: MoveCameraRequest):
    """导演手动标注机位：更新 plot.camera 位置与朝向 → 重绘图纸 → 重算机位视野名单。
    机位是唯一权威：图纸只画这一个机位符号，标注即替换，不再叠加第二个机位。"""
    plot = _normalize_plot(req.plot or {})
    plot["camera"]["x"] = _clamp01(req.x)
    plot["camera"]["y"] = _clamp01(req.y)
    if req.facing is not None:
        plot["camera"]["facing"] = float(req.facing) % 360
        plot["camera"]["movement"] = "固定机位"
        plot["camera"]["path"] = []
    png = render_diagram_png(plot, "topview")
    os.makedirs("static/uploads", exist_ok=True)
    fname = f"diagram_topview_{uuid.uuid4().hex[:10]}.png"
    with open(os.path.join("static", "uploads", fname), "wb") as f:
        f.write(png)
    cam_summary = _camera_view_summary(plot)
    log_prompt("手动移机位", json.dumps(plot.get("camera"), ensure_ascii=False)[:200])
    log_prompt("机位视野计算(移机位后)", cam_summary)
    return {"image_url": f"/static/uploads/{fname}", "plot": plot, "cam_summary": cam_summary}

# ---------- 2.65 调度纵深解读：把俯视口述换算成"人物近/中/远景层次"说明（供分镜图理解远近关系） ----------
class StagingBriefRequest(BaseModel):
    description: str = ""  # 分镜画面描述/提示词
    topview_desc: str | None = None  # 俯视调度图口述（位置关系+机位）
    trajectory_desc: str | None = None  # 轨迹调度图口述（运动轨迹）
    characters: List[Dict[str, Any]] | None = None
    cam_summary: str | None = None  # 程序按坐标计算的机位视野事实（谁入画/谁出画），权威

STAGING_SYSTEM_PROMPT = """你是场面调度解读员。导演给了一个镜头的调度口述（俯视走位/运动轨迹，机位信息含在其中），需要解读成一段"画面纵深调度说明"，让文生图模型准确理解人物的远近关系。

【解读规则】
0. 若提供「机位视野事实」（程序按坐标计算），它关于机位位置与朝向、谁入画/谁不入画、每人身体朝向（面向镜头/背对镜头/侧对镜头面朝画面左或右）的结论全部是权威事实，解读必须逐条原样遵守：不得把名单外的人写进入画、不得把名单内的人写出画面、不得改动机位位置与朝向、不得给任何人另编一个朝向；你只在其基础上细化纵深次序、遮挡关系与画面表现。口述文字与机位视野事实冲突时（如导演手动移过机位后口述仍写旧机位），以机位视野事实为准。
0. 先做坐标系换算：口述里的画左/画右/画上方/画下方是【俯视图平面】上的方位，不是成片画面的方位。换算规则：俯视图上离机位符号近的 = 成片近景，离机位远的 = 成片远景；俯视图的左右按机位朝向换算（当机位位于俯视图下方朝画面上方拍摄时，俯视图左 = 成片画面左、俯视图右 = 成片画面右、俯视图上 = 远景；机位在其它方位时按此几何关系类推，换算依据写明）。
1. 先从口述中确定机位位置与拍摄朝向（如"平拍""固定机位""机位在天台东侧"）；若口述未明确，按最合理的观察点推断并注明（推断）。
2. 机位视角占主导：先以机位与拍摄朝向为准，把出场人物分成两组：
   【镜头后方组】——站在机位所在位置上的人物（视角人物，镜头从他眼中拍出）、以及位于机位身后（与拍摄朝向相反一侧）的人物：一律「在镜头后方，完全出画、不出现」，不标注画面位置；
   【机位前方组】——机位视线前方的一切人物（包括紧挨机位、视野极边缘的人）：按物理透视正常入画，给每人标注【成片画面中的布局】：左右（画面左缘/中部/右缘）+ 与机位的相对远近次序（谁比谁离镜头近、谁最远）与画面表现：
   - 离机位近 → 离镜头近 → 画面中显得大、位于画面下缘，可能部分遮挡别人或被取景框裁切（紧挨机位的人可能只露出腿部/肩膀，按真实视野处理）；
   - 离机位远 → 显得小、靠近画面上缘地平线。
   分寸提示：从分镜描述中识别该镜头的景别与画面主体——主体按景别呈现，但主体同样站在调度图给他安排的位置上取景（不要把主体搬到前景）；其余人物的大小由与镜头的真实距离按透视自然形成，身高一致、纵深相近者大小相近，严禁画成巨人或小矮人；相对远近次序不变，与谁在做什么、谁是主角无关。
3. 左右关系按"站在机位处朝拍摄方向看去"的左右为准；每人朝向必须照抄机位视野事实里的结论（面向镜头＝正脸朝观众、背对镜头＝只见背影、侧对镜头＝按写明的画面左右侧脸），口述里"面朝某人方向"之类的描述只用来理解剧情，与事实冲突时不得写进 brief。
4. 写明明显的前后遮挡关系（谁在谁前面）。
5. 必须写明人物之间的相对间距，保持口述中的比例：谁和谁挨得近（如"小美就在小帅右侧两三步处，两人几乎并排""三人扎堆围在栅栏边"）、谁离人群远（如"瘦猴独自在角落，与人群隔半个画面"）。相近的人在画面中也应相邻、允许互相部分遮挡；不要把人物均匀散布在画面里。
6. 若有移动轨迹，写明从起点纵深到终点纵深的变化（如"从远景跑向近景，逐渐变大"）。
7. 只依据口述与画面描述，不要新增口述里没有的人物或位置；口述含糊处用（推断）标注。
8. 机位视角占主导：机位落在某人物身上时，该人物是视角人物（镜头从他眼中拍出），视角人物与机位身后的人物完全出画；紧挨机位但位于机位视线前方的人物按物理透视正常入画（可为前景，可能只露部分身体）；解读结尾写明谁在镜头后方不入画。取景框是次要参考：人物因物理透视需要入画时（即使在取景框边缘或框外），按机位视角的真实视野处理，不强行因框而删除。

【输出】严格输出 JSON：{"staging_brief": "单段中文，180-320字，可直接拼进文生图提示词"}，不要 markdown，不要其他文字。brief 必须涵盖：① 每个人物离机位的远近次序；② 每个人物的画面左右位置；③ 谁与谁前后相邻或部分遮挡；④ 谁靠近画面下缘/上缘；⑤ 结合该镜头景别判断：哪些人物可能被画面边缘裁切、哪些人物会完全落在取景框外（特写/近景时其余人自然出画，留少量边缘身影由模型自行决定）；主体尽量完整入画。"""

def _fallback_staging_brief(req: StagingBriefRequest) -> str:
    parts = ["先做坐标系换算：口述的画左/画右/上方/下方是俯视平面方位——俯视图上离机位近的人物为前景（占画面大、靠画面下缘），中距为中景，离机位远的人物为背景（占画面小、靠画面上缘地平线）；俯视图的左右按机位朝向换算成成片画面左右（机位在俯视图下方朝上拍时，俯视图左＝成片画面左）；左右关系按从机位朝拍摄方向看去判定；人物之间的间距按口述比例保持，挨得近的在画面中也相邻，不要均匀散开；谁离机位远谁就是背景，与谁在做什么无关"]
    if req.topview_desc:
        parts.append(f"位置口述：{req.topview_desc[:150]}")
    if req.trajectory_desc:
        parts.append(f"轨迹口述：{req.trajectory_desc[:150]}")
    return "；".join(parts)

@app.post("/api/staging-brief")
async def staging_brief(req: StagingBriefRequest):
    """把调度口述解读为纵深调度说明。失败时返回通用换算规则兜底，保证生图流程不中断。"""
    if not (req.topview_desc or "").strip() and not (req.trajectory_desc or "").strip():
        return {"staging_brief": "", "source": "empty"}
    user_lines = [f"镜头画面描述：{req.description or '（无）'}"]
    if req.cam_summary:
        user_lines.append(f"机位视野事实（程序按坐标计算，权威——其中「不入画」名单不可更改，你的解读必须严格遵守）：{req.cam_summary}")
    if req.topview_desc:
        user_lines.append(f"俯视走位口述（含机位）：{req.topview_desc}")
    if req.trajectory_desc:
        user_lines.append(f"运动轨迹口述：{req.trajectory_desc}")
    if req.characters:
        chars = "；".join(f"{str(c.get('name') or '')}" for c in req.characters if c.get("name"))
        if chars:
            user_lines.append(f"出场人物：{chars}")
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": STAGING_SYSTEM_PROMPT},
                {"role": "user", "content": "\n".join(user_lines)},
            ],
            response_format={"type": "json_object"},
            max_tokens=700,
            temperature=0.4,
        )
        data = deepseek_json(response)
        brief = str(data.get("staging_brief") or "").strip()
        if brief:
            log_prompt("调度纵深解读", brief)
            return {"staging_brief": brief, "source": "ai"}
    except Exception as e:
        print(f"[staging-brief] DeepSeek 解读失败，用模板兜底: {type(e).__name__}: {e}")
    fb = _fallback_staging_brief(req)
    log_prompt("调度纵深解读(模板兜底)", fb)
    return {"staging_brief": fb, "source": "template"}

# ---------- 2.7 AU 表情分析：从提示词捕捉表情/动作关键词，映射到 FACS Action Units ----------
class ExtractAURequest(BaseModel):
    prompt: str

@app.post("/api/extract-au")
async def extract_au(req: ExtractAURequest):
    # 第一层：本地词库匹配（零成本、毫秒级）
    result = au_lexicon.extract_aus(req.prompt)
    if result["matches"]:
        result["source"] = "lexicon"
        result["tips"] = au_lexicon.AU_TIPS
        return result

    # 第二层：词库没命中 → DeepSeek 兜底智能映射（拿 AU 速查表当对照依据）
    if getattr(deepseek_client, "api_key", "") not in ("not-configured", None, ""):
        try:
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": (
                        "你是 FACS（面部动作编码系统）表情分析器。给定一段提示词和 AU 速查表，"
                        "找出其中所有表情与动作，映射到 AU 编码组合。规则："
                        f"只允许使用速查表中存在的 AU 编码（含 AU51-AU58 头部动作）；最多输出 5 组；"
                        "提示词中的脸型、年龄、外貌、服饰、身份描述（如鹅蛋脸、圆脸、青年、和服等）一律忽略，"
                        "只提取微表情和面部/头部动作；"
                        'label 用不超过 12 字的中文概括；没有表情/动作就输出 {"matches": []}。'
                        '严格只输出 JSON：{"matches": [{"au": "AU12+AU6", "label": "真心笑"}]}，'
                        "不要 markdown，不要其他文字。"
                    )},
                    {"role": "user", "content": (
                        f"AU速查表：{json.dumps(au_lexicon.AU_TABLE, ensure_ascii=False)}\n\n"
                        f"提示词：{req.prompt}"
                    )},
                ],
                response_format={"type": "json_object"},
                max_tokens=400,
                temperature=0.2,
            )
            data = deepseek_json(response)
            matches = []
            for m in data.get("matches", [])[:5]:
                au = str(m.get("au") or "").strip()
                if re.search(r"AU\d+", au):  # 校验至少含一个合法 AU 编码
                    matches.append({"au": au, "label": str(m.get("label") or "")[:12], "keywords": []})
            if matches:
                codes, seen = [], set()
                for m in matches:
                    for code in re.findall(r"AU\d+", m["au"]):
                        if code not in seen:
                            seen.add(code); codes.append(code)
                log_prompt("AU分析(AI兜底)", req.prompt[:120], f"结果={'/'.join(m['au'] for m in matches)}")
                return {"matches": matches, "au_string": "+".join(codes), "source": "ai"}
        except Exception as e:
            print(f"[AU分析] DeepSeek 兜底失败，按无命中处理: {type(e).__name__}: {e}")

    return {"matches": [], "au_string": "", "source": "lexicon"}

# ---------- 3. 豆包生图 (存本地返回 URL) ----------
MAX_REF_IMAGES = 10  # 豆包 seedream 参考图上限

def select_relevant_references(pool: list[str], meta: list[Dict[str, Any]] | None, context: str) -> list[int]:
    """从候选参考图中筛选与当前画面相关的子集（豆包上限 10 张）。

    策略：
    0. 调度图（kind=diagram，俯视走位/运动轨迹）是导演定稿，永远保留且排在最前，不参与筛选
    1. 总数不超限 → 全部使用（不加延迟）
    2. 超限 → 先按名字出现在上下文中做关键词匹配（零开销）
    3. 关键词匹配后仍超限或匹配为空 → DeepSeek 按语义挑选相关的（角色优先）
    4. 兜底：任何环节失败则取前 10 张（调度图始终保留）
    """
    meta = meta or []
    diagram_idx = [i for i, m in enumerate(meta[:len(pool)]) if str(m.get("kind") or "") == "diagram"]

    def _finalize(picked: list[int]) -> list[int]:
        """合并调度图与筛选结果：调度图排最前，去重、限流、保序"""
        seen, ordered = set(), []
        for i in diagram_idx + list(picked):
            if 0 <= i < len(pool) and i not in seen:
                seen.add(i)
                ordered.append(i)
        return ordered[:MAX_REF_IMAGES]

    if len(pool) <= MAX_REF_IMAGES:
        return _finalize(list(range(len(pool))))

    # 第一步：关键词匹配
    kw_hits = []
    for i, m in enumerate(meta[:len(pool)]):
        if str(m.get("kind") or "") == "diagram":
            continue
        name = str(m.get("name") or "").strip()
        if name and len(name) >= 2 and name in context:
            kw_hits.append(i)
    if 0 < len(kw_hits) <= MAX_REF_IMAGES:
        print(f"[参考图筛选] {len(pool)} 张 -> 关键词命中 {len(kw_hits)} 张: {kw_hits}（调度图固定保留 {len(diagram_idx)} 张）")
        return _finalize(kw_hits)

    # 第二步：DeepSeek 语义精选（调度图已固定保留，只筛其余的）
    try:
        budget = max(MAX_REF_IMAGES - len(diagram_idx), 0)
        candidates = []
        for i in range(min(len(pool), len(meta)) if meta else 0):
            if i in set(diagram_idx):
                continue
            m = meta[i]
            candidates.append({
                "index": i,
                "kind": m.get("kind", ""),
                "name": str(m.get("name") or ""),
                "description": str(m.get("description") or "")[:80],
            })
        if candidates and budget > 0:
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": (
                        "你是分镜参考图筛选器。给定画面描述和候选参考图列表（index/kind/name/description），"
                        "挑选与该画面相关的参考图。规则：只输出 JSON {\"indices\": [..]}；"
                        "最多选 " + str(budget) + " 个；无关的不要选；"
                        "画面中出现某角色时必须选该角色的参考图；场景相关的场景图优先。"
                    )},
                    {"role": "user", "content": f"画面描述：{context}\n\n候选：{json.dumps(candidates, ensure_ascii=False)}"},
                ],
                response_format={"type": "json_object"},
                max_tokens=200,
                temperature=0.1,
            )
            data = deepseek_json(response)
            picked = [int(i) for i in data.get("indices", []) if isinstance(i, (int, str)) and str(i).lstrip("-").isdigit()]
            picked = [i for i in picked if 0 <= i < len(pool)]
            # 去重保序，角色优先排序
            seen = set()
            ordered = []
            for i in sorted(set(picked), key=lambda x: (0 if (meta[x].get("kind") if x < len(meta) else "") == "character" else 1, x)):
                if i not in seen:
                    seen.add(i); ordered.append(i)
            if ordered:
                print(f"[参考图筛选] {len(pool)} 张 -> DeepSeek 精选 {len(ordered[:budget])} 张: {ordered[:budget]}（调度图固定保留 {len(diagram_idx)} 张）")
                return _finalize(ordered[:budget])
    except Exception as e:
        print(f"[参考图筛选] DeepSeek 精选失败，回退前 {MAX_REF_IMAGES} 张: {type(e).__name__}: {e}")

    # 兜底：调度图 + 其余按原顺序取满
    others = [i for i in range(len(pool)) if i not in set(diagram_idx)]
    return _finalize(others[:max(MAX_REF_IMAGES - len(diagram_idx), 0)])

# ---------- 调试日志：把每次发给模型的提示词追加写入文件，便于排查"生成内容不对" ----------
def log_prompt(kind: str, prompt: str, extra: str = ""):
    try:
        from datetime import datetime
        with open("debug_prompts.log", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] {kind} {extra}\n{prompt}\n{'-'*60}\n")
    except Exception:
        pass

# ---------- 导演显式落位检测：重写提示词里直接写了"某人在画面XX"，必须压过程序算的默认落位 ----------
_FRAME_POS_KEYWORDS = [
    # 长词在前，命中多个时取最长（"画面右上角"优先于"画面右"）
    "画面右上角", "画面左上角", "画面右下角", "画面左下角",
    "画面右上方", "画面左上方", "画面右下方", "画面左下方",
    "画面正中间", "画面正中", "画面中央", "画面上方", "画面下方",
    "画面右侧", "画面左侧", "画面右", "画面左",
    "右上角", "左上角", "右下角", "左下角", "正中间", "正中",
    "居中", "中央", "偏右", "偏左", "靠右", "靠左",
]

def _detect_frame_position_overrides(prompt: str, characters: list) -> str:
    """扫描导演最终确认的文字提示词：某句话里同时出现角色名与画面落位词
    （画面右上角/画面左/居中/偏右…）即视为导演对落位的显式指令。
    若不注入覆盖声明，尾部权威名单里的默认"画面左/右"会把导演指令判为冲突作废，
    导致重写永远不生效（生成结果总回到图纸几何位置）。返回"人→落位"清单文本。"""
    import re
    if not prompt:
        return ""
    names = []
    for c in characters or []:
        n = str((c or {}).get("name") or "").strip()
        if n and n in prompt:
            names.append(n)
    # 紧邻落位词前如果是"面向/朝/望向"等词，那是视线或运动方向，不是落位
    _GAZE_PREFIXES = ("面向", "朝向", "望向", "看向", "望着", "走向", "跑向", "扑向",
                      "冲向", "飞向", "朝着", "指", "面朝", "望", "朝", "看")
    hits: list[str] = []

    def _add(entry: str):
        if entry not in hits:
            hits.append(entry)

    for sentence in re.split(r"[。！？\n]", prompt):
        cur_name = None  # 当前句主语：逗号分句没点名时沿用最近提到的人
        # 按逗号/顿号也切开，避免"小帅在画面中央，小美靠左"两句被并成一句错配
        for clause in re.split(r"[；，,、：:]", sentence):
            # 主语识别先行：即使本分句只有视线用法，后面分句仍可沿用主语
            hit_names = [n for n in names if n in clause]
            if hit_names:
                cur_name = hit_names[-1]
            matched = {k for k in _FRAME_POS_KEYWORDS if k in clause}
            # 去掉被同句更长关键词包含的短词（"画面右上角"吃掉"右上角""画面右"）
            kws = [k for k in matched if not any(k != o and k in o for o in matched)]
            for kw in sorted(kws, key=len, reverse=True):
                # 每次出现都检查前缀：整句只有视线用法时跳过
                is_gaze_only = True
                start = 0
                while True:
                    i = clause.find(kw, start)
                    if i < 0:
                        break
                    prefix = clause[max(0, i - 2):i]
                    if not prefix.endswith(_GAZE_PREFIXES):
                        is_gaze_only = False
                    start = i + len(kw)
                if is_gaze_only:
                    continue
                if "主体" in clause or "构图" in clause:
                    # 无角色名的构图指令（如"构图主体略偏右"）按导演指令对待
                    _add("画面主体→" + kw)
                    continue
                if cur_name:
                    _add(f"{cur_name}→{kw}")
    return "；".join(hits[:8])


@app.post("/api/generate-image")
async def generate_image(req: ImageRequest):
    log_prompt("生图", req.prompt, f"size={req.size} 参考图={len(req.reference_urls or [])}张")
    payload = {
        "model": DOUBAO_MODEL,
        "prompt": req.prompt,
        "size": req.size,
        "response_format": "b64_json",
        "watermark": False,
    }
    if req.reference_urls and len(req.reference_urls) > 0:
        # 先按相关性筛选（关键词 -> DeepSeek），再转换；豆包云端无法下载本机 127.0.0.1 URL，
        # 本地图一律转 base64 data URI 直传，公网 URL 原样保留
        context = (req.context or req.prompt or "").strip()
        # 先按机位几何剔除"绝不入画"角色的参考图：生图模型看到角色参考图就会把他画进画面，
        # 提示词里的"绝不入画名单"拗不过参考图，必须在发送前过滤（调度图/场景/道具不受影响）
        pool = list(req.reference_urls or [])
        metas = list(req.reference_meta or [])
        # 空镜保护：提示词声明"空镜头/画面中无人物"时，角色与道具参考图一律不发——
        # 生图模型看到参考图就会把他们画进画面，提示词里的"无人物"拗不过参考图
        empty_shot = any(k in context for k in (
            "空镜", "画面中无人物", "画面里无人物", "无人物", "没有人物", "画面中没有人"))
        vis_names = _visible_character_names(req.plot_data or {})
        if empty_shot or vis_names is not None:
            allowed = set(vis_names) if vis_names is not None else None
            f_pool, f_meta, removed = [], [], []
            for i in range(len(pool)):
                m = metas[i] if i < len(metas) else {}
                kind = str(m.get("kind") or "")
                name = str(m.get("name") or "").strip()
                if empty_shot and kind in ("character", "prop"):
                    removed.append(f"{name or kind}（空镜）")
                    continue
                if kind == "character" and allowed is not None and name and name not in allowed:
                    removed.append(name)
                    continue
                f_pool.append(pool[i]); f_meta.append(m)
            if removed:
                log_prompt("参考图过滤(不入画角色/空镜)",
                           f"剔除：{'、'.join(removed)}"
                           + (f"；保留入画角色：{'、'.join(vis_names) or '无'}" if allowed is not None else ""))
            pool, metas = f_pool, f_meta
        picked = select_relevant_references(pool, metas, context)
        resolved = []
        for i in picked:
            u = pool[i]
            if u.startswith(("http://127.0.0.1", "http://localhost", "/static/")):
                resolved.append(local_url_to_data_uri(u))
            else:
                resolved.append(u)
        payload["image"] = resolved[:MAX_REF_IMAGES]  # 最终安全上限

        # 调度图（俯视走位/运动轨迹）是导演定稿，优先级高于文字提示词：
        # 注入强制约束，让构图布局以调度图为准，文字只负责画风/长相/服装/细节
        # 注意：picked 是过滤后 metas（机位剔除不入画角色后）的索引，不能用原始 reference_meta
        n_diag = sum(
            1 for i in picked
            if i < len(metas) and str(metas[i].get("kind") or "") == "diagram"
        )
        if n_diag:
            # 收集调度图的口述说明（前端把口述内容放在 diagram 参考的 description 里）
            notes = []
            for i in picked:
                if i < len(metas) and str(metas[i].get("kind") or "") == "diagram":
                    d = str(metas[i].get("description") or "").strip()
                    n = str(metas[i].get("name") or "").strip() or "调度图"
                    if d:
                        notes.append(f"▸{n}：{d}")
            directive = (
                f"【机位与调度（最高优先级）】本图的前 {n_diag} 张参考图是导演定稿的调度图纸："
                "正上方垂直俯视的平面图（白底图纸，人偶棋子图标＝人物、旁边标签是人名，「机位」符号＋扇形视锥＝摄影机位置与拍摄朝向，虚线箭头＝运动路线）。"
                "若图中出现红框且框外区域被压暗，红框＝导演画定的取景范围。本镜头的观察点就在机位符号处、朝视锥方向看：\n"
                "规则一·机位定视角：镜头架在「机位」符号处、朝视锥扇形方向拍摄，画面内容＝从机位望出去的景象——"
                "站在机位上的人物是视角人物（镜头从他眼中拍出），视角人物、紧贴机位旁的人物与机位身后的人物一律不出现在画面中；"
                "机位视线前方的人物按真实透视入画——离机位越近画面里显得越大（可为前景，可能只露出腿部、肩膀或部分背影），越远越小、靠近地平线。"
                "特别注意：画面主体的位置由机位视角决定，不因他在文字提示词里是主角就把镜头架到他那边去拍。\n"
                "规则二·图纸定位置：人物在画面中的左右按「站在机位处朝拍摄方向看」换算（机位视线左侧＝画面左侧，不得机械套用图纸方位）；"
                "人物之间的间距与前后遮挡按图纸比例，挨得近的画得挨近，禁止均匀散布。"
                "人物的位置与相对远近由图纸锁定；每个人的动作、表情、长相、服装、所持道具由文字提示词决定——调度只安排在哪里，不改变任何人在做什么。\n"
                "规则三·景别定取景：景别、主体、构图、画风按文字提示词。景别只是取景框的远近：特写/近景＝取景框只框住主体所在区域，"
                "其他人自然落在框外看不见，严禁把他们挪到主体旁边、缩小塞进画面或为露脸改变位置；"
                "全景时所有人按规则一的透视呈现。同一地面人物身高一致，纵深相近者大小相近，严禁巨人/小矮人；"
                "严禁为了让某人入画或完整而挪动位置、违例缩放；处在取景框边缘的人可被画面裁切。\n"
                "规则四·红框约束：红框只是导演画定的取景范围（画面边缘裁切参考），不是机位——镜头位置与朝向只由「机位」符号决定，"
                "严禁把红框理解为镜头所在处、或朝红框方向拍摄。画面内容严格限制在红框内——压暗区域里的人与景一律不画（即使文字提示词提到了他们）；"
                "红框与机位真实视野冲突时，以机位与真实透视为准，红框只作构图参考。\n"
                "注意：最终输出的是正常视角的真实镜头画面，不要画成俯视图、示意图或图纸形式；画面里的人物是成片形象，不是图标。\n"
            )
            if notes:
                directive += "导演口述的调度要求（注意：机位位置与朝向、谁入画、每人身体朝向一律以上面图纸坐标数据为准——导演手动移过机位后口述里的旧机位描述作废；口述只用来补充动作、剧情与细节）：\n" + "\n".join(notes) + "\n"
            else:
                directive += "请观察调度图纸中人偶图标的站位与朝向、虚线运动路线、机位符号与视锥方向，把上述空间关系落到画面里。\n"
            if req.diagram_brief:
                directive += f"【画面纵深调度解读】（机位视角换算的辅助参考）：{req.diagram_brief.strip()}\n"
            # 坐标翻译：生图模型看不懂俯视图纸上的图标，但听得懂"谁离镜头最近/在哪侧/谁不入画"
            plot_txt = format_plot_to_prompt(req.plot_data or {})
            if plot_txt:
                directive += "【调度图纸坐标数据（程序按坐标翻译，权威）】\n" + plot_txt + "\n"
            # 导演在重写框里直接指定的画面落位（画面右上角/偏右/居中…）必须压过程序算的默认落位——
            # 否则权威名单里的"画面左/右"会把导演指令判为冲突作废，重写永远不生效
            override_txt = _detect_frame_position_overrides(
                req.prompt, (req.plot_data or {}).get("characters") or [])
            if override_txt:
                directive += (
                    f"【导演落位指令·最高优先级】导演已在本条提示词中直接指定画面落位：{override_txt}。"
                    "这是导演的显式构图指令，人物在成片画面中的实际落位以导演指定为准；"
                    "上面名单与纵深解读中的「画面左/右」只是按调度图纸几何推算的默认落位，"
                    "与导演指定冲突时按导演指定执行，不算文字冲突、不作废。"
                    "但谁入画谁不入画、人物身体朝向、远近大小关系仍按上面名单与透视规则执行——导演落位指令只改变人物在画面中的方位。\n"
                )
            directive += (
                "【结尾重申·最高优先级】请逐条核对后再出图：①镜头位置与朝向只由「机位」符号决定；"
                "②画面中出现的人物必须且只能是入画名单里的人，绝不入画名单里的人一个都不能出现；"
                "③人物远近与大小按入画名单从近到远按透视排列；"
                "④若存在【导演落位指令】，被指定人物在画面中的落位按导演指定执行；"
                "⑤最终输出正常视角的真实镜头画面，不是俯视图或示意图。\n"
            )
            # 🌟 强指令放到 prompt 末尾：大模型对末尾指令的服从度最高，放开头容易被当背景介绍忽略
            payload["prompt"] = payload["prompt"] + "\n\n" + directive
            log_prompt("调度约束注入(尾部)", f"调度图 {n_diag} 张\n" + "\n".join(notes)
                       + f"\nbrief={req.diagram_brief or ''}\nplot_data={'有' if req.plot_data else '无'}\n---\n" + directive, "")

    headers = {
        "Authorization": f"Bearer {DOUBAO_API_KEY}",
        "Content-Type": "application/json",
    }
    
    t0 = __import__("time").time()
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            resp = await client.post(
                f"{DOUBAO_BASE}/images/generations",
                headers=headers,
                json=payload,
            )
        except httpx.HTTPError as e:
            # 超时/断网：云端可能已生成并计费，但本地拿不到图——记下来便于和方舟账单对账
            log_prompt("生图超时/中断(云端可能已计费)", f"{type(e).__name__}: {e}", f"耗时{__import__('time').time()-t0:.0f}s")
            raise HTTPException(504, f"生图请求超时或中断（云端可能已计费）：{type(e).__name__}，请稍后再试")

        if resp.status_code != 200:
            print("\n❌ 豆包 API 报错详情:", resp.text, "\n")
            log_prompt("生图失败", resp.text[:300], f"HTTP {resp.status_code} 耗时{__import__('time').time()-t0:.0f}s")
            raise HTTPException(resp.status_code, f"豆包生图失败：{resp.text}")

        data = resp.json()
        b64_data = data["data"][0]["b64_json"]
        # 记录豆包实际返回几张（确认是否存在"生成了多张但只显示一张"的情况）
        print(f"🖼️ 豆包本次返回图片数: {len(data.get('data', []))}")
        log_prompt("生图成功", f"返回{len(data.get('data', []))}张，耗时{__import__('time').time()-t0:.0f}s")
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

    # 运动轨迹指令：调度图纸（俯视/轨迹）坐标翻译成自然语言，与参考图一起约束视频的人物走位与镜头运动
    prompt = req.prompt
    if req.plot_data:
        plot_txt = format_plot_to_prompt(req.plot_data)
        if plot_txt:
            prompt = (f"{req.prompt}\n\n【运动轨迹指令】（由导演定稿的调度图纸坐标翻译而来，"
                      f"人物走位、运动快慢、身体朝向与镜头运动必须严格遵守）\n{plot_txt}\n"
                      "【身份唯一规则】画面中每个角色全程只有一个身体、一份存在：上述走位是画面中已有角色的连续位移，"
                      "不是新角色入场；严禁出现复制人、分身、双胞胎——比如出现两个憨憨相向而行、一个走近镜头另一个还站在原地，都是错误；"
                      "角色从画面外走入时，走进来的就是他本人，画面里原本的他平滑移动过来即可。"
                      "白底俯视调度图纸类参考图仅用于理解走位方向与机位关系，"
                      "严禁把图纸本身、白底平面图、人偶棋子图标、扇形视锥、虚线箭头画进视频画面。")

    target_workflow = req.workflow_id if req.workflow_id else AUTODL_WORKFLOW

    # ---------- 按参考图宽高比自动选横/竖屏档位 ----------
    # H3 系工作流默认档是竖屏（如 768p竖=768×1344），横屏参考图不改档位就会输出竖屏视频。
    # AutoDL 工作流详情接口（/api/v1/comfyui/workflows/{id}）的 input_rules 里有 resolution
    # 参数的官方枚举（如 "480p竖/480p横/480p(1:1)/768p竖/768p横/768p(1:1)"），据此选档最可靠。
    _workflow_res_cache: dict = {}
    async def get_resolution_options(client, headers):
        if target_workflow in _workflow_res_cache:
            return _workflow_res_cache[target_workflow]
        try:
            r = await client.get(f"{AUTODL_BASE}/api/v1/comfyui/workflows/{target_workflow}", headers=headers)
            data = (r.json() or {}).get("data") or {}
            rules = data.get("input_rules") or {}
            options = [o.get("label") for o in ((rules.get("resolution") or {}).get("options") or []) if o.get("label")]
            print(f"📋 工作流 {target_workflow} 支持 resolution 档位: {options or '（无此参数，用默认档）'}")
        except Exception as e:
            print(f"⚠️ 查询工作流 {target_workflow} 参数失败（不影响提交）: {type(e).__name__}: {e}")
            options = []
        _workflow_res_cache[target_workflow] = options
        return options

    def pick_resolution(options: list, aspect) -> str | None:
        """横图选"横"档，竖图选"竖"档，方图选"(1:1)"档；同方向优先 768p，其次 480p。"""
        if not options:
            return None
        want = None
        if aspect:
            if aspect > 1.05: want = "横"
            elif aspect < 0.95: want = "竖"
            else: want = "(1:1)"
        if want:
            for tier in ("768p", "480p"):
                for o in options:
                    if tier in o and want in o:
                        return o
            for o in options:  # 方向命中但档位不全时，取该方向任意档
                if want in o:
                    return o
        # 拿不到宽高比时回退 768p竖（与工作流默认一致，避免误选 480p 降画质）
        for o in options:
            if "768p" in o:
                return o
        return options[0]

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

        def build_payload(put_first: bool, put_last: bool, resolution: str | None = None) -> dict:
            p = {"prompt": prompt, "duration": req.duration}
            if frame_mode == "reference":
                for i, url in enumerate(req.image_urls[:9]):
                    p[f"ref_image_{i}"] = to_val(url)
            else:
                if put_first and req.first_frame_url:
                    p["first_frame"] = to_val(req.first_frame_url)
                if put_last and req.last_frame_url:
                    p["last_frame"] = to_val(req.last_frame_url)
            # resolution 档位（含横竖屏）：仅在查询到工作流支持该参数时才发送；
            # zm 系列老工作流不接受 resolution（传了报 RequestParameterIsWrong），保持默认档
            if resolution:
                p["resolution"] = resolution
            return p

        target_workflow = req.workflow_id if req.workflow_id else AUTODL_WORKFLOW
        res_options = await get_resolution_options(client, headers)
        chosen_res = pick_resolution(res_options, req.image_aspect)
        if chosen_res:
            print(f"🖼️ 参考图宽高比={req.image_aspect} → 自动选择 resolution 档位: {chosen_res}")

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
            payload = build_payload(False, False, chosen_res)
        elif frame_mode == "first":
            payload = build_payload(True, False, chosen_res)   # 只传首帧
        elif frame_mode == "last":
            payload = build_payload(False, True, chosen_res)   # 只传尾帧
        else:
            payload = build_payload(True, True, chosen_res)    # 首尾帧过渡
        print(f"📤 提交给 AutoDL: workflow={target_workflow}, 模式={frame_mode}, 图数量={len(req.image_urls[:9])}, resolution={chosen_res or '默认'}")
        print(f"📝 视频提示词: {prompt[:200]}{'…' if len(prompt) > 200 else ''}")
        log_prompt("生视频", prompt, f"workflow={target_workflow} 模式={frame_mode} duration={req.duration} resolution={chosen_res or '默认'} plot_data={'有' if req.plot_data else '无'}")

        resp = await submit(payload)
        data = resp.json() if resp.status_code == 200 else {}
        # 该工作流若不接受 resolution 参数（RequestParameterIsWrong 且提到 resolution），去掉后原样重试
        if chosen_res and data.get("code") == "RequestParameterIsWrong" and "resolution" in (data.get("msg") or ""):
            print("🔄 工作流不接受 resolution 参数，去掉该参数重试（使用工作流默认档位）")
            chosen_res = None
            payload = build_payload(frame_mode in ("first", "first_last"), frame_mode in ("last", "first_last"))
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
            payload = build_payload(True, True, chosen_res)
            resp = await submit(payload)
            data = resp.json() if resp.status_code == 200 else {}
        # 兜底：若参数名不被接受（报未定义参数），改用 ref_image_0 再试一次
        elif frame_mode in ("first", "last", "first_last") and data.get("code") == "RequestParameterIsWrong" \
                and "未定义的参数" in (data.get("msg") or ""):
            print("🔄 first_frame/last_frame 参数名不被接受，改用 ref_image_0 重试")
            payload = {"prompt": prompt, "duration": req.duration}
            if chosen_res:
                payload["resolution"] = chosen_res
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

# ---------- 项目导出/导入：项目 JSON + 引用到的 static/uploads 媒体文件打包成 zip ----------
class ExportProjectRequest(BaseModel):
    name: str = "未命名项目"
    data: Dict[str, Any]

def _collect_upload_refs(obj: Any, found: set) -> None:
    """递归收集项目数据里引用的所有 /static/uploads/ 文件名。"""
    if isinstance(obj, str):
        for m in re.findall(r"/static/uploads/([A-Za-z0-9_.\-]+)", obj):
            found.add(m)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_upload_refs(v, found)
    elif isinstance(obj, list):
        for v in obj:
            _collect_upload_refs(v, found)

@app.post("/api/export-project")
async def export_project(req: ExportProjectRequest):
    blob = json.dumps(req.data, ensure_ascii=False)
    found: set = set()
    _collect_upload_refs(req.data, found)
    packed = 0
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project.json", blob)
        for name in sorted(found):
            # 防路径穿越：只要纯文件名
            safe_name = os.path.basename(name)
            p = os.path.join("static", "uploads", safe_name)
            if safe_name and os.path.isfile(p):
                zf.write(p, f"uploads/{safe_name}")
                packed += 1
    buf.seek(0)
    safe_stem = re.sub(r'[\\/:*?"<>|\s]+', "_", req.name).strip("_") or "project"
    filename = f"{safe_stem}.aianime.zip"
    print(f"📤 导出项目「{req.name}」：媒体文件 {packed} 个，包大小 {buf.getbuffer().nbytes / 1024 / 1024:.1f} MB")
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=\"export.aianime.zip\"; filename*=UTF-8''{quote(filename)}"},
    )

@app.post("/api/import-project")
async def import_project(file: UploadFile = File(...)):
    try:
        zf = zipfile.ZipFile(io.BytesIO(await file.read()))
    except zipfile.BadZipFile:
        raise HTTPException(400, "不是有效的项目导出包（应为 .aianime.zip）")
    with zf:
        names = zf.namelist()
        if "project.json" not in names:
            raise HTTPException(400, "导出包缺少 project.json，无法导入")
        try:
            project = json.loads(zf.read("project.json").decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise HTTPException(400, "project.json 解析失败，导出包可能已损坏")
        if not isinstance(project, dict):
            raise HTTPException(400, "project.json 结构异常，无法导入")
        restored = 0
        for n in names:
            if n.startswith("uploads/") and not n.endswith("/"):
                fn = os.path.basename(n)  # 防路径穿越，只取纯文件名
                if not fn or fn != n.split("/", 1)[1]:
                    continue
                dst = os.path.join("static", "uploads", fn)
                if not os.path.exists(dst):  # 同名文件已存在则跳过，不覆盖
                    with open(dst, "wb") as f:
                        f.write(zf.read(n))
                    restored += 1
    print(f"📥 导入项目「{project.get('name', '')}」：恢复媒体文件 {restored} 个")
    return {"project": project, "restored_files": restored}

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