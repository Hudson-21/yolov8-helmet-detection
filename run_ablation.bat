@echo off
chcp 65001 >nul
REM ============================================================
REM  消融实验排队脚本：自动等 exp5_cbam 跑完，再依次训 baseline 与 SE
REM  直接双击运行即可；中途想停就关掉窗口（已完成的实验不会丢）
REM  进度日志：logs\ablation.log
REM ============================================================
cd /d E:\YOLO\safety_helmet
set PY=D:\Anaconda3\envs\yolov8\python.exe
set LOG=logs\ablation.log
set EPOCHS=25

if not exist logs mkdir logs
echo. >> %LOG%
echo ========== 排队开始 %date% %time% ========== >> %LOG%

REM ---------- 1) 等待 exp5_cbam 真正训练结束（最多等 3.5 小时） ----------
REM 判定依据是训练结束才生成的 logs\train_exp5_cbam.json。
REM 不要数 results.csv 行数：断点续训会重跑一轮，行数会重复导致提前触发。
set /a WAITED=0
:wait_loop
if exist logs\train_exp5_cbam.json goto start_exp6
echo [%time%] 等待 exp5_cbam 训练结束标记… (%WAITED% 分钟) >> %LOG%
if %WAITED% GEQ 210 goto timeout_exit
set /a WAITED+=1
timeout /t 60 /nobreak >nul
goto wait_loop

:timeout_exit
echo [%time%] 等待超时（3.5 小时），exp5_cbam 仍未结束，退出 >> %LOG%
echo 等待超时，请稍后重新运行本脚本
pause
exit /b 1

REM ---------- 2) 基线：原生 YOLOv8n 结构 ----------
:start_exp6
echo. >> %LOG%
echo ========== %date% %time% 开始训练 baseline (原生结构) ========== >> %LOG%
"%PY%" src\train.py --model baseline --name exp6_baseline --hyp configs\hyp_tuned.yaml --epochs %EPOCHS% --imgsz 640 --batch 16 --workers 6 --no-default-alias >> %LOG% 2>&1
echo [%time%] baseline 结束，退出码 %errorlevel% >> %LOG%

REM ---------- 3) 对照：SE 通道注意力 ----------
echo. >> %LOG%
echo ========== %date% %time% 开始训练 SE (通道注意力) ========== >> %LOG%
"%PY%" src\train.py --model se --name exp7_se --hyp configs\hyp_tuned.yaml --epochs %EPOCHS% --imgsz 640 --batch 16 --workers 6 --no-default-alias >> %LOG% 2>&1
echo [%time%] SE 结束，退出码 %errorlevel% >> %LOG%

REM ---------- 4) 三组一起评估 ----------
echo. >> %LOG%
echo ========== %date% %time% 开始评估三组权重 ========== >> %LOG%
"%PY%" src\val.py --weights weights\trained\exp5_cbam_best.pt weights\trained\exp6_baseline_best.pt weights\trained\exp7_se_best.pt >> %LOG% 2>&1

echo. >> %LOG%
echo ========== 全部完成 %date% %time% ========== >> %LOG%
echo.
echo 全部跑完了。结果见：
echo   runs\exp6_baseline\  runs\exp7_se\
echo   results\eval_val.md   （三组对比表）
echo   logs\ablation.log     （完整过程日志）
pause
