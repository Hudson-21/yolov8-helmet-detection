"""模型评估 —— 输出 mAP / Precision / Recall / FPS，并生成可直接写进文档的表格。

评分项对应：精度指标 9 分（mAP@0.5 ≥ 0.85 满分）、推理速度 3 分（FPS ≥ 15）。

用法：
    python src/val.py                                    # 评估默认 best.pt
    python src/val.py --weights weights/trained/exp_base_best.pt --split val
    python src/val.py --weights a.pt --compare b.pt       # 多模型横向对比（消融实验用）
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ultralytics import YOLO

from config import CLASS_NAMES, DATA_YAML, DEVICE, IMG_SIZE, LOGS_DIR, RESULTS_DIR, ensure_dirs, resolve_weights


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="评估模型精度与速度")
    p.add_argument("--weights", nargs="+", default=None, help="一个或多个权重文件")
    p.add_argument("--data", type=Path, default=DATA_YAML)
    p.add_argument("--split", default="val", choices=["val", "test", "train"])
    p.add_argument("--imgsz", type=int, default=IMG_SIZE)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--conf", type=float, default=0.001, help="验证用低阈值，保证 PR 曲线完整（业界惯例）")
    p.add_argument("--compare", nargs="*", default=None, help="额外参与对比的权重")
    return p.parse_args()


def benchmark_fps(weights: Path, imgsz: int, device, rounds: int = 30) -> float:
    """用随机张量做纯推理测速，返回 FPS（不含前后处理，反映模型本身速度）。"""
    from ultralytics import YOLO as _YOLO

    model = _YOLO(str(weights))
    net = model.model.to("cuda" if str(device) != "cpu" and torch.cuda.is_available() else "cpu").eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz, device=next(net.parameters()).device)
    with torch.no_grad():
        for _ in range(5):  # 预热
            net(dummy)
        if dummy.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(rounds):
            net(dummy)
        if dummy.is_cuda:
            torch.cuda.synchronize()
        cost = (time.perf_counter() - t0) / rounds
    return 1.0 / cost if cost > 0 else 0.0


def evaluate(weights: Path, args: argparse.Namespace) -> dict:
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        plots=True,
        verbose=False,
    )
    box = metrics.box
    fps = benchmark_fps(weights, args.imgsz, args.device)
    per_class = {}
    for i, name in CLASS_NAMES.items():
        try:
            per_class[name] = {
                "precision": float(box.p[i]),
                "recall": float(box.r[i]),
                "mAP50": float(box.ap50[i]),
                "mAP50-95": float(box.ap[i]),
            }
        except (IndexError, TypeError):
            continue
    return {
        "weights": str(weights),
        "split": args.split,
        "imgsz": args.imgsz,
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "fps_pure_inference": round(fps, 1),
        "inference_ms": round(1000.0 / max(fps, 1e-6), 2),
        "per_class": per_class,
    }


def main() -> None:
    args = parse_args()
    ensure_dirs()
    weights_list = [Path(w) for w in (args.weights or [resolve_weights()])]
    if args.compare:
        weights_list += [Path(w) for w in args.compare]
    for w in weights_list:
        if not w.exists():
            raise SystemExit(f"权重不存在: {w}")

    results = []
    for w in weights_list:
        print(f"\n>>> 评估 {w.name}")
        r = evaluate(w, args)
        results.append(r)
        print(
            f"    mAP50={r['mAP50']:.4f}  mAP50-95={r['mAP50-95']:.4f}  "
            f"P={r['precision']:.4f}  R={r['recall']:.4f}  "
            f"FPS={r['fps_pure_inference']:.1f}"
        )

    # ---------------- 输出对比表 ----------------
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = LOGS_DIR / f"eval_{args.split}.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# 模型评估结果（{args.split} 集，imgsz={args.imgsz}）",
        "",
        "| 权重 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | FPS |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {Path(r['weights']).name} | {r['mAP50']:.4f} | {r['mAP50-95']:.4f} | "
            f"{r['precision']:.4f} | {r['recall']:.4f} | {r['fps_pure_inference']:.1f} |"
        )
    if results and results[0]["per_class"]:
        lines += ["", "## 各类别指标（第一个权重）", "", "| 类别 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |", "|---|---|---|---|---|"]
        for name, m in results[0]["per_class"].items():
            lines.append(
                f"| {name} | {m['precision']:.4f} | {m['recall']:.4f} | {m['mAP50']:.4f} | {m['mAP50-95']:.4f} |"
            )
    md_path = RESULTS_DIR / f"eval_{args.split}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n结果明细: {json_path}")
    print(f"对比表格: {md_path}  （可直接粘到项目文档）")
    best = max(results, key=lambda r: r["mAP50"])
    if best["mAP50"] >= 0.85:
        print(f"✓ 最优权重 {Path(best['weights']).name} 的 mAP@0.5={best['mAP50']:.4f}，达到评分满分线 0.85")
    else:
        print(f"! 当前 mAP@0.5={best['mAP50']:.4f} 未到 0.85，可加数据 / 调 imgsz / 换更大模型")


if __name__ == "__main__":
    main()
