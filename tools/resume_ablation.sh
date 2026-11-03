#!/bin/bash
# 续跑 baseline 与 SE（workers 已降到 2，避开上次的内存 OOM），然后三组一起评估
PY=/d/Anaconda3/envs/yolov8/python.exe
cd /e/YOLO/safety_helmet || exit 1
LOG=logs/ablation2.log
mkdir -p logs
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

say "===== 续跑开始 ====="

say "===== 续训 baseline（从第 11 轮接）====="
$PY src/train.py --resume runs/exp6_baseline/weights/last.pt --no-default-alias >> "$LOG" 2>&1
say "baseline 退出码 $?"
sleep 25   # 等 dataloader 子进程彻底退出，避免与下一个任务的 worker 叠加占内存

say "===== 续训 SE（从第 3 轮接）====="
$PY src/train.py --resume runs/exp7_se/weights/last.pt --no-default-alias >> "$LOG" 2>&1
say "SE 退出码 $?"
sleep 25

say "===== 三组一起评估（val + test）====="
$PY src/val.py --weights weights/trained/exp5_cbam_best.pt weights/trained/exp6_baseline_best.pt weights/trained/exp7_se_best.pt --split val >> "$LOG" 2>&1
say "val 评估退出码 $?"
$PY src/val.py --weights weights/trained/exp5_cbam_best.pt weights/trained/exp6_baseline_best.pt weights/trained/exp7_se_best.pt --split test >> "$LOG" 2>&1
say "test 评估退出码 $?"

say "===== 全部完成 ====="
