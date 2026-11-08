# -*- coding: utf-8 -*-
"""已废弃：风格库配图不再从 Wikimedia Commons 抓取。

原因：风格库里的「3D卡通 / 日式赛璐璐 / 吉卜力 / 新海诚 / 赛博朋克」都是当代创作风格，
公有领域图库里没有对应素材，按关键词抓回来的图（雕塑、老照片、无关绘画）
和风格名对不上，已全部替换。

现在请用：
    python make_style_previews.py             # 只补缺失
    python make_style_previews.py --force     # 全部重生成
    python make_style_previews.py --only pixel,ink_wash --force

它会调用工作台自己的豆包 seedream 接口，用「同一画面主体 + 不同画风描述」
生成 static/style-previews/*.jpg，风格可比对且不会跑题。
"""
import sys

print(__doc__)
sys.exit("fetch_style_images.py 已废弃，请改用 make_style_previews.py")
