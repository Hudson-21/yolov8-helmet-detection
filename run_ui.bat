@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo   安全帽与反光衣检测系统 - 图形界面启动
echo ============================================
set PY=D:\Anaconda3\envs\yolov8\python.exe
if not exist "%PY%" (
    echo [错误] 未找到 Python 环境: %PY%
    echo 请先按 README.md 第三章搭建环境，或修改本文件里的 PY 路径。
    pause
    exit /b 1
)
"%PY%" src\ui_app.py
if errorlevel 1 (
    echo.
    echo [程序异常退出] 请把上面的报错信息截图反馈
    pause
)
