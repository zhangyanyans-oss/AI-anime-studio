# -*- coding: utf-8 -*-
"""AU 表情对照词库与关键词提取器。

AU = Facial Action Coding System (FACS) Action Unit，面部动作编码系统的标准单元。
用途：从生图/生视频提示词中智能捕捉表情与动作关键词，映射到标准 AU 编码，
可用于：表情一致性检查、跨镜头表情对齐、表情驱动工具（Live2D / Audio2Face /
ComfyUI 表情控制节点）的参数参考。

提取策略：
1. 本地词库子串匹配（零成本、毫秒级，覆盖中英文常用表达）
2. 词库未命中时由 main.py 调 DeepSeek 兜底智能映射（用下方 AU_TABLE 当对照表）
"""

# ---------- 单个 AU 速查表（人类可读对照表，也是 AI 兜底映射的依据） ----------
AU_TABLE = {
    "AU1":  "内眉上抬（担忧、悲伤的开端）",
    "AU2":  "外眉上抬（惊讶、挑眉）",
    "AU4":  "眉毛下压/皱眉（生气、专注、难过）",
    "AU5":  "上眼睑抬高（瞪眼、惊讶）",
    "AU6":  "脸颊上抬（真笑的标志，眼角挤出鱼尾纹）",
    "AU7":  "眼睑收紧（眯眼、怀疑、审视）",
    "AU9":  "鼻子皱起（厌恶）",
    "AU10": "上唇上抬（厌恶、狞笑）",
    "AU12": "嘴角上拉（微笑的核心单元）",
    "AU14": "嘴角收紧出现酒窝（冷笑、轻蔑、尴尬陪笑）",
    "AU15": "嘴角下拉（悲伤、撇嘴、失望）",
    "AU17": "下巴上抬（撅嘴、强忍情绪）",
    "AU18": "嘴唇前撅（嘟嘴、要亲亲）",
    "AU20": "嘴唇横向拉伸绷直（恐惧、紧张）",
    "AU22": "嘴唇撮成筒状（吹口哨、呸）",
    "AU23": "嘴唇收紧（咬牙、压抑愤怒）",
    "AU24": "嘴唇压紧（抿嘴、忍耐、尴尬）",
    "AU25": "双唇分开（张嘴、说话）",
    "AU26": "下巴放下、张颌（惊讶、大喊）",
    "AU27": "嘴大张拉伸（怒吼、尖叫、打哈欠）",
    "AU28": "嘴唇吮吸（咬唇、紧张）",
    "AU38/39": "鼻孔扩张/收缩（暴怒、极度恐惧的前兆）",
    "AU43": "闭眼（闭目、熟睡、放空）",
    "AU45": "眨眼（瞬目、使眼色）",
    "AU53": "头上仰（抬头、傲慢、看天）",
    "AU54": "头低垂（低头、沮丧、羞愧）",
    "AU55/56": "头左倾/右倾（歪头）",
    "AU57/58": "头左转/右转（回头、张望）",
}

# ---------- 表情/动作关键词 → AU 组合词库 ----------
# keywords 命中方式：对提示词做不区分大小写的子串匹配（中英文通用）
AU_SETS = [
    # --- 笑 ---
    {"au": "AU12", "label": "微笑",
     "keywords": ["微笑", "浅笑", "含笑", "嘴角上扬", "嘴角微微上扬", "笑意", "smile", "smiling", "slight smile"]},
    {"au": "AU6+AU12+AU25", "label": "开心大笑/真心笑（眼角带笑才算真笑）",
     "keywords": ["大笑", "开心", "高兴", "快乐", "灿烂的笑容", "开怀", "眼睛弯成月牙", "笑得很", "laugh", "laughing", "joy", "happy", "cheerful", "grin", "beaming"]},
    {"au": "AU12+AU6", "label": "温柔微笑",
     "keywords": ["温柔地笑", "温柔微笑", "温柔的笑", "温暖的笑", "gentle smile", "warm smile", "soft smile"]},
    {"au": "AU12(单侧)+AU14", "label": "冷笑/坏笑（文章：颧大肌单侧收缩）",
     "keywords": ["冷笑", "坏笑", "狞笑", "狡黠地笑", "冷冷一笑", "不屑冷笑", "smirk", "sneer"]},
    {"au": "AU12+AU53", "label": "得意（笑+抬头）",
     "keywords": ["得意", "扬扬得意", "沾沾自喜", "骄傲地笑", "smug", "triumphant"]},
    {"au": "AU14+AU24", "label": "尴尬陪笑",
     "keywords": ["尴尬", "窘迫", "干笑", "讪笑", "awkward", "embarrassed"]},
    {"au": "AU12(单侧)", "label": "窃笑/憋笑",
     "keywords": ["窃笑", "偷笑", "憋笑", "嘴角一勾", "snicker", "giggle", "snigger"]},
    {"au": "AU15+AU12(单侧)", "label": "苦笑/自嘲",
     "keywords": ["苦笑", "自嘲", "无奈地笑", "自嘲地", "bitter smile", "self-mocking", "wry smile"]},
    {"au": "AU12(单侧)+AU7", "label": "轻佻/挑逗",
     "keywords": ["轻佻", "挑逗", "调戏", "挑眉坏笑", "媚眼", "flirty", "teasing", "seductive", "playful smirk"]},
    {"au": "AU12+AU6+AU55/AU56", "label": "宠溺（柔笑+微歪头）",
     "keywords": ["宠溺", "慈爱", "疼爱", "温柔地注视", "doting", "affectionate", "loving gaze", "fondly"]},
    # --- 眉眼 ---
    {"au": "AU4", "label": "皱眉",
     "keywords": ["皱眉", "眉头紧锁", "眉心拧", "愁眉", "眉头紧蹙", "frown", "furrowed brow", "knit brows"]},
    {"au": "AU1+AU2", "label": "挑眉",
     "keywords": ["挑眉", "扬起眉毛", "眉梢上挑", "扬眉", "raise eyebrows", "raised brow"]},
    {"au": "AU6+AU7", "label": "眯眼",
     "keywords": ["眯起眼睛", "眯眼", "眯着眼睛", "眯缝着眼", "squint", "narrowed eyes"]},
    {"au": "AU5", "label": "瞪大眼睛",
     "keywords": ["瞪大眼睛", "睁大眼睛", "圆睁双眼", "眼睛瞪得", "wide-eyed", "eyes wide open"]},
    {"au": "AU7+AU4", "label": "审视/怀疑的眼神",
     "keywords": ["怀疑", "审视", "狐疑", "警惕地看", "戒备", "suspicious", "wary", "skeptical", "distrustful"]},
    {"au": "AU43", "label": "闭眼",
     "keywords": ["闭上眼睛", "闭眼", "闭目", "双眼紧闭", "eyes closed", "closed eyes"]},
    {"au": "AU45", "label": "眨眼/使眼色",
     "keywords": ["眨眼", "眨了眨", "wink", "blinking", "blinks"]},
    {"au": "AU43+AU54", "label": "熟睡/昏睡",
     "keywords": ["睡着", "熟睡", "昏睡", "沉睡", "asleep", "sleeping"]},
    {"au": "AU43(半闭)+AU45", "label": "困倦/睡眼惺忪",
     "keywords": ["困倦", "昏昏欲睡", "睡眼惺忪", "眼皮打架", "没精打采地眨眼", "drowsy", "sleepy", "droopy eyelids", "dozing"]},
    {"au": "AU27+AU43", "label": "打哈欠",
     "keywords": ["打哈欠", "哈欠", "呵欠", "yawn", "yawning"]},
    {"au": "AU43(半闭)+AU25", "label": "如释重负（眉头舒展+呼气）",
     "keywords": ["如释重负", "松了口气", "长舒一口气", "终于放心", "relief", "relieved"]},
    # --- 惊讶/恐惧 ---
    {"au": "AU1+AU2+AU5+AU26", "label": "惊讶（眉抬+瞪眼+张颌）",
     "keywords": ["惊讶", "吃惊", "惊愕", "意外", "难以置信", "惊呆", "surprise", "surprised", "astonished", "amazed"]},
    {"au": "AU1+AU2+AU5+AU25+AU27", "label": "震惊（文章级全开：惊到张大嘴）",
     "keywords": ["震惊", "目瞪口呆", "呆住", "下巴都掉了", "惊掉下巴", "stunned", "dumbfounded", "gobsmacked", "jaw dropped"]},
    {"au": "AU1+AU2+AU4", "label": "恐慌前兆/不安（恐惧眉）",
     "keywords": ["不安", "忐忑", "心里发毛", "隐隐的担忧", "预感不妙", "unease", "apprehension", "anxious", "dread"]},
    {"au": "AU38/AU39", "label": "鼻孔扩张（暴怒/极度恐惧前兆）",
     "keywords": ["鼻孔扩张", "鼻翼扇动", "鼻翼翕动", "nostrils flared", "nostril flare"]},
    {"au": "AU4+AU10+AU25+AU27", "label": "爆发大吼",
     "keywords": ["大吼", "怒吼", "咆哮", "嘶吼", "怒骂", "roar", "screaming", "howling"]},
    {"au": "AU4+AU5+AU20+AU25", "label": "恐惧/惊恐",
     "keywords": ["恐惧", "害怕", "惊恐", "吓得", "惊慌", "战栗", "fear", "terrified", "scared", "panic", "dread"]},
    # --- 悲伤 ---
    {"au": "AU1+AU4+AU15", "label": "悲伤",
     "keywords": ["悲伤", "难过", "伤心", "哀伤", "忧伤", "委屈", "低落", "sad", "sorrow", "grief", "heartbroken", "melancholy"]},
    {"au": "AU1+AU4+AU15+AU17+AU25", "label": "哭泣",
     "keywords": ["哭泣", "泪流满面", "含泪", "泪", "哽咽", "抽泣", "cry", "crying", "tears", "weeping", "sob"]},
    {"au": "AU1+AU2+AU7+AU15", "label": "含泪强忍（眼周收紧憋泪）",
     "keywords": ["强忍泪水", "含泪强忍", "眼眶湿润忍住", "忍住眼泪", "holding back tears", "teary-eyed"]},
    {"au": "AU15+AU54", "label": "沮丧/失望（撇嘴+低头）",
     "keywords": ["沮丧", "失望", "泄气", "垂头丧气", "意志消沉", "disappointed", "dejected", "depressed", "defeated"]},
    # --- 愤怒 ---
    {"au": "AU4+AU5+AU17+AU23", "label": "愤怒（皱眉+瞪眼+咬紧）",
     "keywords": ["愤怒", "生气", "恼火", "怒视", "暴怒", "青筋", "气得", "angry", "anger", "furious", "rage", "glare", "outraged"]},
    {"au": "AU23+AU24", "label": "压抑怒气（咬牙/抿嘴）",
     "keywords": ["咬牙", "咬紧牙关", "抿紧嘴唇", "抿着嘴", "强忍", "压抑着怒", "clenched jaw", "gritted teeth", "pressed lips"]},
    # --- 厌恶/轻蔑 ---
    {"au": "AU9+AU10+AU17", "label": "厌恶",
     "keywords": ["厌恶", "嫌弃", "恶心", "反感", "作呕", "鄙夷地皱鼻", "disgust", "disgusted", "revolted"]},
    {"au": "AU12+AU14+AU53", "label": "轻蔑/不屑",
     "keywords": ["轻蔑", "不屑", "鄙视", "瞧不起", "傲慢", "倨傲", "contempt", "scorn", "disdain", "haughty"]},
    # --- 嘴部动作 ---
    {"au": "AU25+AU26", "label": "张嘴/张大嘴",
     "keywords": ["张开嘴", "张嘴", "张大嘴巴", "张大了嘴", "open mouth", "opened mouth", "agape", "jaw drop"]},
    {"au": "AU25", "label": "说话/喊叫",
     "keywords": ["说着", "开口", "喊道", "大喊", "呼喊", "喊叫", "低声说", "speaking", "shouting", "yelling", "talking", "murmur"]},
    {"au": "AU18", "label": "嘟嘴/撅嘴",
     "keywords": ["嘟嘴", "嘟着嘴", "撅嘴", "撅着嘴", "鼓起腮", "pout", "pouting"]},
    {"au": "AU28", "label": "咬唇",
     "keywords": ["咬唇", "咬着嘴唇", "咬住下唇", "bite lip", "biting lip"]},
    # --- 头部动作 ---
    {"au": "AU53", "label": "抬头/仰头",
     "keywords": ["抬头", "仰头", "昂首", "仰起脸", "head up", "looking up", "tilt head back"]},
    {"au": "AU54", "label": "低头/垂头",
     "keywords": ["低头", "垂头", "耷拉着脑袋", "低下头", "head down", "looking down", "bowed head"]},
    {"au": "AU55/AU56", "label": "歪头（左右由画面决定）",
     "keywords": ["歪头", "偏着头", "歪着脑袋", "歪了歪头", "head tilt", "tilting head"]},
    {"au": "AU57/AU58", "label": "转头/回头张望",
     "keywords": ["回头", "转头", "扭头", "回望", "turn head", "looking back", "glance back"]},
    {"au": "头部动作（无标准AU）", "label": "点头/摇头",
     "keywords": ["点头", "摇头", "颔首", "nod", "shake head"]},
    # --- 特殊状态 ---
    {"au": "AU6+AU12", "label": "害羞（羞笑+泛红）",
     "keywords": ["害羞", "脸红", "羞涩", "羞怯", "shy", "blushing", "bashful", "flustered"]},
    {"au": "无AU激活", "label": "面无表情/冷漠（脸部肌肉放松）",
     "keywords": ["面无表情", "冷漠", "无表情", "呆滞", "放空", "淡漠", "expressionless", "blank", "numb", "stoic"]},
]

# ---------- 微表情控制小贴士（参考《80个表情包AI提示词》方法论，已剔除脸型/外貌内容） ----------
AU_TIPS = [
    "克制写法：加 subtle / slight / barely 可抑制卡通化崩坏（实测提升稳定性）",
    "单段聚焦：一条提示词最多 3-5 个肌群、AU 不超过 6 个",
    "真笑看眼角：只有 AU12 没有 AU6 的笑容易假，眼角鱼尾纹是真心笑的标志",
    "情绪强度靠时间窗口拆：0-0.5s 起势、0.5-1.2s 到峰值，每窗 2-4 秒",
    "AU 编码仅作表情分析参考，豆包/视频模型不直接识别，可用于人工校对或表情驱动工具",
]

# AU 编码提取（用于汇总 au_string）
import re
_AU_CODE_RE = re.compile(r"AU\d+")

# 叙事基调短语（描述故事风格，不是面部表情），匹配前先剔除，避免 style_bible 里的
# "自嘲式幽默" 之类写法让所有提示词都误命中同一个表情
_TONE_STRIP_RE = re.compile(r"自嘲[式的]?(?:幽默|喜感|喜剧|段子|调侃|玩笑)")


def extract_aus(text: str) -> dict:
    """从提示词中提取表情/动作关键词并映射到 AU。

    返回:
        {
            "matches": [{"au": "AU12+AU6", "label": "...", "keywords": ["大笑", ...]}, ...],
            "au_string": "AU12+AU6+AU25",   # 去重汇总，可直接给表情驱动工具用
            "source": "lexicon",
        }
    """
    if not text or not text.strip():
        return {"matches": [], "au_string": "", "source": "lexicon"}

    # 先剔除叙事基调短语（如 style_bible 里的"自嘲式幽默"），再做表情匹配
    t = _TONE_STRIP_RE.sub("", text.lower())
    hits = []
    for item in AU_SETS:
        matched = [kw for kw in item["keywords"] if kw.lower() in t]
        if matched:
            hits.append({
                "au": item["au"],
                "label": item["label"],
                "keywords": matched[:4],  # 展示前几个命中词即可
                "_score": len(matched),
            })

    # 命中词多的排前面，最多展示 8 组，避免列表爆炸
    hits.sort(key=lambda x: -x["_score"])
    hits = hits[:8]
    for h in hits:
        h.pop("_score", None)

    # 汇总去重的标准 AU 编码（"头部动作（无标准AU）"等说明性条目不参与汇总）
    codes, seen = [], set()
    for h in hits:
        for code in _AU_CODE_RE.findall(h["au"]):
            if code not in seen:
                seen.add(code)
                codes.append(code)

    return {"matches": hits, "au_string": "+".join(codes), "source": "lexicon"}
