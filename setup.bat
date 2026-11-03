@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo   环境安装（已配清华镜像源）
echo ============================================
set PY=D:\Anaconda3\envs\yolov8\python.exe
if not exist "%PY%" (
    echo [错误] 未找到 conda 环境 yolov8
    echo 请先执行: conda create -n yolov8 --clone py313
    pause
    exit /b 1
)

echo.
echo [1/3] 安装 requirements.txt 依赖 ...
"%PY%" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 goto fail

echo.
echo [2/3] 以可编辑模式安装改过的 Ultralytics 源码 ...
"%PY%" -m pip install -e E:\YOLO\ultralytics-main -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 goto fail

echo.
echo [3/3] 自检 ...
"%PY%" -c "from ultralytics import YOLO; m=YOLO('yolov8n-cbam.yaml'); print('CBAM 模型构建成功'); import torch; print('CUDA 可用:', torch.cuda.is_available())"
if errorlevel 1 goto fail

echo.
echo ============================================
echo   安装完成，双击 run_ui.bat 启动界面
echo ============================================
pause
exit /b 0

:fail
echo.
echo [失败] 上一步命令返回错误，请检查网络或依赖冲突
pause
exit /b 1
