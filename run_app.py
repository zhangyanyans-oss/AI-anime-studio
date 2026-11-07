"""AI 动画工作台 - exe 打包入口。

双击运行：自动选一个空闲端口启动服务，并打开默认浏览器。
关闭本窗口（或 Ctrl+C）即退出服务。
"""
import os
import socket
import sys
import threading
import time
import webbrowser

# Windows GBK 控制台兼容：强制 UTF-8 输出，避免 emoji 报 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import uvicorn
from main import app, clean_temp_files


def _pick_port(start=8000):
    """从 8000 开始找一个空闲端口，避免端口冲突起不来。"""
    port = start
    for _ in range(20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1
    return start


def _open_browser(url):
    time.sleep(1.5)  # 等服务起来
    try:
        webbrowser.open(url)
    except Exception:
        print(f"请手动打开浏览器访问: {url}")


if __name__ == "__main__":
    port = _pick_port()
    url = f"http://127.0.0.1:{port}"
    print(f"\n🎬 AI 动画工作台启动中… 浏览器将自动打开 {url}")
    print("   关闭本窗口即退出服务。\n")

    try:
        clean_temp_files()  # 启动时清理一次过期临时文件
    except Exception as e:
        print(f"清理临时文件跳过: {e}")

    threading.Thread(target=_open_browser, args=(url,), daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
