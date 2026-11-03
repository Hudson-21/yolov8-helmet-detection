#!/bin/bash
# 消融实验排队：等 exp5_cbam 真正训练结束 → 训 baseline → 训 SE → 三组一起评估
#
# 完成判定用 logs/train_exp5_cbam.json 是否存在，而不是数 results.csv 行数：
# 断点续训会重跑一轮已完成的 epoch，行数会重复，数行数会提前触发。
# train.py 只在整个训练流程走完时才写这个 json，是可靠的完成信号。
PY=/d/Anaconda3/envs/yolov8/python.exe
cd /e/YOLO/safety_helmet || exit 1
LOG=logs/ablation.log
EPOCHS=25
DONE_MARK=logs/train_exp5_cbam.json
mkdir -p logs
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

say "===== 排队启动（等待 exp5_cbam 训练结束标记）====="
waited=0
while true; do
  if [ -f "$DONE_MARK" ]; then say "检测到 $DONE_MARK，exp5 已训练结束，开始对照实验"; break; fi
  if [ "$waited" -ge 200 ]; then say "等待超时（约 3.3 小时）仍未见完成标记，退出"; exit 1; fi
  waited=$((waited+1)); sleep 60
done

say "===== 开始 baseline（原生 YOLOv8n）====="
$PY src/train.py --model baseline --name exp6_baseline --hyp configs/hyp_tuned.yaml --epochs $EPOCHS --imgsz 640 --batch 16 --workers 6 --no-default-alias >> "$LOG" 2>&1
say "baseline 结束，退出码 $?"

say "===== 开始 SE（通道注意力）====="
$PY src/train.py --model se --name exp7_se --hyp configs/hyp_tuned.yaml --epochs $EPOCHS --imgsz 640 --batch 16 --workers 6 --no-default-alias >> "$LOG" 2>&1
say "SE 结束，退出码 $?"

say "===== 三组权重一起评估 ====="
$PY src/val.py --weights weights/trained/exp5_cbam_best.pt weights/trained/exp6_baseline_best.pt weights/trained/exp7_se_best.pt >> "$LOG" 2>&1
say "===== 全部完成 ====="
