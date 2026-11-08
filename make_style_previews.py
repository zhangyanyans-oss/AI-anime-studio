# -*- coding: utf-8 -*-
"""风格库配图生成器（用工作台自己的豆包 seedream 出图）。

为什么不用图库抓取：风格库里的「皮克斯 3D / 日式赛璐璐 / 新海诚光影 / 赛博朋克」
都是当代创作风格，公有领域图库里根本没有对应的图，抓回来必然驴唇不对马嘴。
现在改成「一段画风描述 + 同一个画面主体」逐张生成，37 张可以横向比对，
也不会出现「说是水墨、其实是一张照片」的情况。

用法:
    python make_style_previews.py                 # 只补缺失的
    python make_style_previews.py --force         # 全部重生成
    python make_style_previews.py --only pixel,ink_wash --force
输出:
    static/style-previews/<slug>.jpg  (1200x675, 16:9，与卡片裁切一致)

注意：提示词里的画风描述不要写「某某动画截图」「复刻某作品」，
       豆包会判定为版权风险直接拒图；用可视觉化的画风词最稳。
"""
import argparse
import asyncio
import base64
import io
import os
import time

import httpx
from dotenv import load_dotenv
from PIL import Image

load_dotenv(os.path.join(os.getcwd(), ".env"))

OUT = os.path.join("static", "style-previews")
SIZE = "1920x1080"          # 直接出 16:9
FINAL = (1200, 675)
QUALITY = 86
CONCURRENCY = 3

# 所有图共用的主体：同一人物同一构图，方便横向比较画风
SUBJECT = "一位短发少女，穿米白色连衣裙，侧身回头看向镜头，微微笑着"
DEFAULT_SETTING = "背景是山坡、远山与小镇屋顶"

# (slug, 画风描述, 场景覆盖)
# 场景覆盖只给「画风即场景」的类别，其余一律沿用 DEFAULT_SETTING，
# 让同一构图贯穿 37 张，风格差异才看得出来。
STYLES = [
    ("neutral", "写实摄影，自然光，真实的皮肤与衣料质感，清晰通透", None),
    ("3d_cartoon", "3D 卡通动画电影质感，圆润造型，大眼睛，柔和次表面散射，明亮饱和色块，全局光照", None),
    ("cinematic_3d", "3A 游戏 CG 过场：精细的三维角色建模，PBR 材质，次表面散射的皮肤，渲染器出图，电影布光，中景带环境的构图", None),
    ("guofeng_3d", "国风 3D 动画：三维建模的立体人物，丝绸与金属材质高光，敦煌金绿色调，体积光，CG 渲染质感明显", None),
    ("clay", "黏土定格动画：整个画面都是橡皮泥捏成的，人物、衣服、草地和房屋都是黏土材质，保留手工捏痕与指纹，微缩布景打光", None),
    ("3d_thickpaint", "3D 渲染的立体人物，表面覆盖明显的厚涂笔触与颜料肌理，像被油画笔刷扫过的三维模型，笔触方向清晰，光影浑厚", None),
    ("felt", "羊毛毡手工：整幅画面是毛毡做的，人物、衣服、草地和房屋都由羊毛毡缝制，像毛毡玩偶而非真人，纤维毛边清晰，缝线可见", None),
    ("cel_anime", "动漫赛璐璐风格，平涂色块，粗线稿，硬边阴影，色彩干净", None),
    ("ghibli", "手绘动画，水彩厚涂的背景，温暖自然光，细腻的空气感，柔和的绿色调", None),
    ("shinkai", "绚丽晚霞与积云，强烈逆光与镜头光斑，超饱和色彩，清爽的青春感", None),
    ("retro_anime", "90 年代复古日漫，赛璐璐平涂，胶片颗粒，轻微色偏，高饱和复古配色", None),
    ("thick_anime", "二次元厚涂插画，精致的五官刻画，丰富的色彩层次，厚涂笔触", None),
    ("us_comic", "美式漫画：欧美漫画的硬朗五官与比例，粗黑描边，网点阴影，平涂大色块，超级英雄漫画的分镜感", None),
    ("shoujo", "少女漫画，柔和粉彩，网点与闪光，花瓣装饰，梦幻唯美", None),
    ("dark_anime", "暗黑系动漫插画：动漫化的人物五官，粗黑线稿，强对比的深色阴影与冷调高光，哥特式压抑氛围，细密排线", None),
    ("ink_wash", "水墨写意，宣纸，纯黑白灰，毛笔笔触与飞白，墨色浓淡，留白", None),
    ("blue_green", "青绿山水绢本设色：整幅画用石青石绿矿物颜料绘制，人物也是工笔勾线后平涂，绢本暖黄底色", None),
    ("gongbi", "工笔重彩：清晰细密的墨线轮廓，石青石绿与朱砂的重彩平涂，人物面部白皙，衣纹线条工整精致", None),
    ("dunhuang", "敦煌壁画，斑驳的土黄墙面，石绿与赭石，飞天式线描，岁月剥落的痕迹", None),
    ("papercut", "中国剪纸，红色镂空，平面化的民间艺术造型，刀刻的硬朗边缘", None),
    ("bluewhite", "青花瓷：整幅画面是白底钴蓝的青花瓷纹样，人物与风景都用钴蓝线条与晕染描绘，缠枝莲纹装饰，釉面光泽", None),
    ("film35", "35mm 胶片摄影：粗重的胶片颗粒与噪点，暖橙色调，轻微柔焦与暗角，边缘有胶片划痕", None),
    ("cyberpunk", "赛博朋克，霓虹雨夜，全息广告牌，湿滑反光的地面，青紫高对比",
     "背景是霓虹闪烁的赛博都市街道，巨大的全息广告牌，湿漉漉的路面映着灯"),
    ("wasteland", "末日废土美学，沙尘弥漫，残破建筑与锈蚀金属，低饱和的土黄与灰",
     "背景是沙尘笼罩的废墟，坍塌的建筑与锈蚀钢架，天空昏黄"),
    ("noir", "黑色电影，高对比黑白，百叶窗的条纹光影，烟雾，悬疑气氛",
     "背景是夜间的旧办公室，百叶窗把光切成一条条，烟雾浮在光柱里"),
    ("ice", "冰雪末世，冷蓝低饱和，浓雾与冰晶，苍凉的静默感",
     "背景是冰封的荒原与蓝色冰川，浓雾弥漫"),
    ("vaporwave", "蒸汽波风格：粉紫渐变天空，棕榈树剪影，荧光网格地平线，复古 80 年代电脑图形，VHS 噪点",
     "背景是粉紫渐变的天空、棕榈树剪影与荧光网格地平线"),
    ("horror", "暗黑恐怖，单一光源，浓重阴影，颗粒噪点，低明度",
     "背景是阴森废弃的木质走廊，唯一光源是一盏摇曳的烛火，大部分画面沉在黑暗里"),
    ("family_warm", "温暖生活剧质感，柔和的暖黄光，浅景深，自然的生活气息",
     "背景是傍晚的温馨客厅，暖黄台灯与布艺沙发"),
    ("gothic", "哥特式暗黑华丽，尖拱与蕾丝，彩窗的幽光，抑郁而优雅",
     "背景是哥特式教堂内部，尖拱、彩绘玻璃与石柱，光线幽暗华美"),
    ("vangogh", "梵高式厚涂油画，流动的短笔触，漩涡状的星空，浓烈的黄蓝对比", None),
    ("watercolor_kids", "儿童绘本水彩，边缘柔和晕开，纸张纹理，天真明亮的配色", None),
    ("flat", "扁平矢量插画，几何化的色块，无描边，简洁现代的构成", None),
    ("pixel", "8-bit 像素游戏画面，块状色块，无渐变硬边，复古 FC 调色板，低分辨率像素格清晰", None),
    ("ukiyo", "浮世绘木刻版画，平涂色块，鲜明的轮廓线，波浪与云纹图式，和纸质感", None),
    ("line_min", "极简单线插画，一根连续的线，大量留白，黑白两色，克制优雅", None),
    ("retro_print", "复古印刷海报，套色错位与粗糙纸纹，50 年代广告的配色与版式", None),
]


def build_prompt(style_prompt: str, setting):
    # 画风放最前面：豆包对「先给的风格锚点」服从度明显更高
    return f"{style_prompt}。{SUBJECT}。{setting or DEFAULT_SETTING}。"


async def gen_one(client: httpx.AsyncClient, slug: str, prompt: str, force: bool):
    dest = os.path.join(OUT, slug + ".jpg")
    if not force and os.path.exists(dest) and os.path.getsize(dest) > 20000:
        return slug, "skip"
    last = None
    for attempt in range(3):
        try:
            r = await client.post(
                f"{os.getenv('DOUBAO_BASE', 'https://ark.cn-beijing.volces.com/api/v3')}/images/generations",
                headers={"Authorization": f"Bearer {os.getenv('ARK_API_KEY')}",
                         "Content-Type": "application/json"},
                json={"model": os.getenv("DOUBAO_MODEL", "doubao-seedream-5-0-flash-260915"),
                      "prompt": prompt, "size": SIZE,
                      "response_format": "b64_json", "watermark": False},
            )
            if r.status_code == 200:
                raw = base64.b64decode(r.json()["data"][0]["b64_json"])
                im = Image.open(io.BytesIO(raw)).convert("RGB")
                if im.size != FINAL:
                    im = im.resize(FINAL, Image.LANCZOS)
                im.save(dest, "JPEG", quality=QUALITY, optimize=True)
                return slug, f"ok {os.path.getsize(dest)//1024}KB"
            last = f"HTTP {r.status_code} {r.text[:160]}"
            if r.status_code not in (429, 500, 502, 503, 504):
                break      # 4xx（含内容审核）重试也不会过，直接放弃
        except Exception as e:  # 网络抖动 / 超时
            last = f"{type(e).__name__}: {e}"
        await asyncio.sleep(2 * (attempt + 1))
    return slug, "FAIL " + str(last)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    todo = [s for s in STYLES if not only or s[0] in only]
    if only:
        missing = only - {s[0] for s in STYLES}
        if missing:
            raise SystemExit("未知 slug: " + ", ".join(sorted(missing)))

    sem = asyncio.Semaphore(CONCURRENCY)
    done = 0
    failed = []

    async with httpx.AsyncClient(timeout=240) as client:
        async def worker(item):
            nonlocal done
            slug, sp, setting = item
            async with sem:
                t0 = time.time()
                _, res = await gen_one(client, slug, build_prompt(sp, setting), args.force)
            done += 1
            if res.startswith("FAIL"):
                failed.append(slug)
            print(f"[{done:2d}/{len(todo)}] {slug:16s} {res}  ({time.time()-t0:.0f}s)", flush=True)

        await asyncio.gather(*(worker(s) for s in todo))

    print("\n失败:", failed or "无")


if __name__ == "__main__":
    asyncio.run(main())
