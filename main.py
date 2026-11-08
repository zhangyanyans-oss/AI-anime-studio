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
10. 输出必须是可直接 json.loads 的合法 JSON：双引号、无注释、无尾逗号，字符串内不要出现真实换行。"""

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

【三类提示词的固定规范】
1. 角色（character）：角色设定图。style_bible 原文 + 「单人全身立绘、完整入画（头顶到脚踝全部可见）、正面或四分之三侧面、自然站姿目视镜头、简洁纯色浅背景、柔和均匀打光」+ 逐一写死可复用的外观锚点：发型发色、瞳色、五官特征、体型年龄、服装款式与颜色、随身标志物。60-120 字。禁止出现场景、动作戏、其他人物。
2. 场景（scene）：场景设定空镜。style_bible 原文 + 大远景空镜（画面中无人物）+ 空间布局与 2-3 个标志性陈设 + 时间天气 + 光线方向。60-120 字。
3. 分镜（storyboard）：结构化模板，单行输出，小节用【】标题依次排列，每个小节必须写：【画面内容】在(地点)，(谁)正在(做什么)，画面正在发生(具体事件)，出现的角色逐字复用其角色锚点的外貌服装描述，写明表情与动作瞬间；【镜头细节】焦段(如35mm/85mm)、构图(主体居中/对称/前景遮挡)、景别(远/全/中/近/特)、机位(固定/低角度/俯拍/仰拍)；【美术风格】style_bible 原文；【光源】光源类型(顶光/逆光/侧光/蝴蝶光/伦勃朗光/硬光/柔光/自然光等)及光如何影响场景；【氛围】观众的情绪感受；【影调】冷暖色调、饱和度、明暗关系；【约束】正向表达：无文字水印分格、动作自然符合物理规律、本镜头需锁定的要点。上下文不足写【待补：说明】，禁止编造。

【硬性禁止】画面出现任何文字、字幕、水印、边框、分格、多格拼图；禁止抽象词（如“幸福”“回忆”“命运”）；禁止把台词写进画面。

【video_prompt 规范】仅分镜类型需要输出，单行输出：【动作时序】首先…然后…最后…(总时长=该镜头持续秒数)；【镜头运动】推/拉/摇/移/环绕/跟随/固定机位；【声音】仅当有明确声音需求时注明，没有就省略。画面起点即该分镜图。硬性禁止：引入分镜图中不存在的新人物/物体/场景；禁止“突然”“闪现”“切换到”等跳变词；不要复述静态外观。

【用户修改要求】旧提示词是用户当前工作框内的内容，是最高优先级的内容来源：其中用户亲笔写下的具体细节（人物年龄、体型、五官、服装、随身物、动作、表情、地点、道具、光源、氛围、风格等）必须逐字保留，严禁替换、省略、泛化或“换种说法”。重写只允许做三件事：①把内容套进对应类型的结构化模板并分节；②补全模板要求但用户没写的维度（不确定的写【待补：说明】，禁止编造）；③语言规范化。若角色锚点与用户描述冲突，一律以用户描述为准；若旧提示词很短且明显是一句修改指令（如“换成雨夜”）而非完整提示词，将其整体视为用户的新要求，结合上下文重写出完整的结构化提示词。

【待补占位规则】若旧提示词含有【待补：xx】，而用户的新要求没有给出对应信息，则原样保留该【待补：xx】（可以写得更具体），不要替用户拍板编造；用户在新要求里明确了的内容，必须替换掉对应的【待补】。
【素材与资产】素材引用（参考图/音频/视频）与全局设定（资产库）是创作者的输入，不由你生成、不要虚构；相关需求写进【约束】或【待补】。

【输出】严格输出 JSON：{"image_prompt": "...", "video_prompt": "..."}。非分镜类型 video_prompt 输出空字符串。不要 markdown 代码块，不要任何 JSON 之外的文字。"""

@app.post("/api/regenerate-prompt")
async def regenerate_prompt(req: RegeneratePromptRequest):
    if getattr(deepseek_client, "api_key", "") in ("not-configured", None, ""):
        raise HTTPException(400, "未配置 DeepSeek API Key，无法重新生成提示词。")
    t = (req.type or "").lower()
    if t not in ("character", "scene", "storyboard"):
        raise HTTPException(400, "type 必须是 character / scene / storyboard")

    type_label = {"character": "角色设定图", "scene": "场景设定空镜", "storyboard": "分镜图"}[t]
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

# ---------- 2.6 AU 表情分析：从提示词捕捉表情/动作关键词，映射到 FACS Action Units ----------
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
    1. 总数不超限 → 全部使用（不加延迟）
    2. 超限 → 先按名字出现在上下文中做关键词匹配（零开销）
    3. 关键词匹配后仍超限或匹配为空 → DeepSeek 按语义挑选相关的（角色优先）
    4. 兜底：任何环节失败则取前 10 张
    """
    if len(pool) <= MAX_REF_IMAGES:
        return list(range(len(pool)))

    meta = meta or []
    # 第一步：关键词匹配
    kw_hits = []
    for i, m in enumerate(meta[:len(pool)]):
        name = str(m.get("name") or "").strip()
        if name and len(name) >= 2 and name in context:
            kw_hits.append(i)
    if 0 < len(kw_hits) <= MAX_REF_IMAGES:
        print(f"[参考图筛选] {len(pool)} 张 -> 关键词命中 {len(kw_hits)} 张: {kw_hits}")
        return kw_hits

    # 第二步：DeepSeek 语义精选
    try:
        candidates = []
        for i in range(min(len(pool), len(meta)) if meta else 0):
            m = meta[i]
            candidates.append({
                "index": i,
                "kind": m.get("kind", ""),
                "name": str(m.get("name") or ""),
                "description": str(m.get("description") or "")[:80],
            })
        if candidates:
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": (
                        "你是分镜参考图筛选器。给定画面描述和候选参考图列表（index/kind/name/description），"
                        "挑选与该画面相关的参考图。规则：只输出 JSON {\"indices\": [..]}；"
                        "最多选 " + str(MAX_REF_IMAGES) + " 个；无关的不要选；"
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
            seen, ordered = set(), []
            for i in sorted(set(picked), key=lambda x: (0 if meta[x].get("kind") == "character" else 1, x)):
                if i not in seen:
                    seen.add(i); ordered.append(i)
            if ordered:
                print(f"[参考图筛选] {len(pool)} 张 -> DeepSeek 精选 {len(ordered[:MAX_REF_IMAGES])} 张: {ordered[:MAX_REF_IMAGES]}")
                return ordered[:MAX_REF_IMAGES]
    except Exception as e:
        print(f"[参考图筛选] DeepSeek 精选失败，回退前 {MAX_REF_IMAGES} 张: {type(e).__name__}: {e}")

    return list(range(MAX_REF_IMAGES))

# ---------- 调试日志：把每次发给模型的提示词追加写入文件，便于排查"生成内容不对" ----------
def log_prompt(kind: str, prompt: str, extra: str = ""):
    try:
        from datetime import datetime
        with open("debug_prompts.log", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] {kind} {extra}\n{prompt}\n{'-'*60}\n")
    except Exception:
        pass

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
        picked = select_relevant_references(req.reference_urls, req.reference_meta, context)
        resolved = []
        for i in picked:
            u = req.reference_urls[i]
            if u.startswith(("http://127.0.0.1", "http://localhost", "/static/")):
                resolved.append(local_url_to_data_uri(u))
            else:
                resolved.append(u)
        payload["image"] = resolved[:MAX_REF_IMAGES]  # 最终安全上限

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
            p = {"prompt": req.prompt, "duration": req.duration}
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
        print(f"📝 视频提示词: {req.prompt[:200]}{'…' if len(req.prompt) > 200 else ''}")
        log_prompt("生视频", req.prompt, f"workflow={target_workflow} 模式={frame_mode} duration={req.duration} resolution={chosen_res or '默认'}")

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
            payload = {"prompt": req.prompt, "duration": req.duration}
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