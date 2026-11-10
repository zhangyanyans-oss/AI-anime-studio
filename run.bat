@echo off
title AI 动画工作台启动器
cd /d "%~dp0"

echo =========================================
echo       AI 动画工作台 - 正在启动
echo =========================================
echo.

:: 1. 找一个可用的 Python（优先 3.13/3.11 等成熟版本；
::    过新的版本可能缺少 pydantic 等包的预编译轮子导致装依赖失败）
set "PYCMD="
py -3.13 -c "import sys" >nul 2>&1 && set "PYCMD=py -3.13"
if not defined PYCMD py -3.12 -c "import sys" >nul 2>&1 && set "PYCMD=py -3.12"
if not defined PYCMD py -3.11 -c "import sys" >nul 2>&1 && set "PYCMD=py -3.11"
if not defined PYCMD py -3 -c "import sys" >nul 2>&1 && set "PYCMD=py -3"
if not defined PYCMD python -c "import sys" >nul 2>&1 && set "PYCMD=python"
if not defined PYCMD (
    echo [错误] 未找到 Python，请先安装 Python 3.11 或更高版本。
    echo 下载地址: https://www.python.org/downloads/
    echo 安装时务必勾选 "Add python.exe to PATH"。
    echo.
    pause
    exit /b 1
)
echo [0/3] 使用 Python: %PYCMD%
echo.

:: 2. 虚拟环境不存在则自动创建（首次运行约几十秒）
if not exist "venv\Scripts\activate.bat" (
    echo [1/3] 首次运行，正在创建虚拟环境...
    %PYCMD% -m venv venv
    if errorlevel 1 (
        echo [错误] 虚拟环境创建失败，请截图本窗口的报错信息反馈。
        echo.
        pause
        exit /b 1
    )
)

:: 3. 激活虚拟环境（此后 python 都指向 venv 内的解释器）
call venv\Scripts\activate.bat

:: 4. 依赖不齐就自动安装（关键包可导入视为依赖齐全）
python -c "import uvicorn, fastapi, httpx, dotenv, PIL, multipart, openai" >nul 2>&1
if errorlevel 1 (
    echo [2/3] 正在安装依赖，首次约 1-2 分钟，请耐心等待...
    python -m pip install -q -r requirements.txt
    if errorlevel 1 (
        echo 默认源安装失败，改用清华镜像重试...
        python -m pip install -q -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
        if errorlevel 1 (
            echo [错误] 依赖安装失败，请检查网络后重新运行本程序。
            echo.
            pause
            exit /b 1
        )
    )
)

:: 5. 启动服务
echo [3/3] 正在启动后端服务，请勿关闭此窗口...
echo.
echo     ^> 启动成功后，请在浏览器打开: http://127.0.0.1:8000
echo     ^> 按 Ctrl+C 可以停止服务
echo.
echo =========================================

:: 启动服务（--reload 修改代码后自动重启；用 python -m 方式调用更可靠）
python -m uvicorn main:app --reload

:: 服务崩溃或被关闭时暂停，以便看清报错信息
echo.
echo 服务已退出。如果上方有报错信息，请截图反馈。
pause
