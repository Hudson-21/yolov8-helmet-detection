"""模型导出 —— 把 .pt 转成 ONNX，用于"轻量化 / 部署拓展"加分项。

用法：
    python src/export.py                      # 默认导出 best.pt → ONNX
    python src/export.py --format onnx --imgsz 640 --half
    python src/export.py --weights weights/trained/exp_cbam_best.pt --format onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from config import IMG_SIZE, TRAINED_DIR, resolve_weights


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="导出模型")
    p.add_argument("--weights", default=None)
    p.add_argument("--format", default="onnx", choices=["onnx", "torchscript", "engine", "openvino"])
    p.add_argument("--imgsz", type=int, default=IMG_SIZE)
    p.add_argument("--half", action="store_true", help="FP16 导出（GPU 部署用，进一步提速）")
    p.add_argument("--dynamic", action="store_true", help="动态 batch / 尺寸")
    p.add_argument("--simplify", action="store_true", default=True, help="导出后做图简化")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights) if args.weights else resolve_weights()
    print(f"导出权重: {weights}")
    model = YOLO(str(weights))
    out = model.export(
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        dynamic=args.dynamic,
        simplify=args.simplify,
    )
    out_path = Path(out)
    target = TRAINED_DIR / out_path.name
    if out_path.exists() and out_path.parent != TRAINED_DIR:
        import shutil

        TRAINED_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_path, target)
        out_path = target
    size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0
    print(f"导出完成: {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
