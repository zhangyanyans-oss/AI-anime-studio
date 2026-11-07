@echo off
chcp 65001 >nul
title AI 动画工作台启动器
echo =========================================
echo       🎬 AI 动画工作台 - 正在启动
echo =========================================
echo.

:: 1. 检查虚拟环境是否存在
if not exist "venv\Scripts\activate.bat" (
    echo [错误] 未找到虚拟环境 "venv" 文件夹！
    echo 请先在当前目录打开命令行，输入：python -m venv venv
    echo 然后激活并安装依赖。
    pause
    exit /b
)

:: 2. 激活虚拟环境
echo [1/2] 正在激活虚拟环境...
call venv\Scripts\activate.bat

:: 3. 启动 Uvicorn 服务
echo [2/2] 正在启动后端服务，请勿关闭此窗口...
echo.
echo     ^> 启动成功后，请在浏览器打开: http://127.0.0.1:8000
echo     ^> 按 Ctrl+C 可以停止服务
echo.
echo =========================================

:: 启动服务（使用 --reload 可以在你修改代码后自动重启）
uvicorn main:app --reload

:: 如果服务崩溃或被关闭，暂停一下以便看清报错信息
pause