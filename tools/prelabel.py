"""预标注工具 —— 用 COCO 预训练权重把"人员"框先画好，减少手工标注量。

为什么只预标 person：
    yolov8n.pt 是在 COCO 上训练的，COCO 里有 person 类（id 0），所以人员框可以
    直接复用，精度相当高；但安全帽（helmet）和反光衣（vest）**不在 COCO 里**，
    没有任何现成权重能标，只能人工画。而这两类恰恰是项目的核心和难点。

所以本工具的价值是：把"每个人都要手动画一个大框"这件事自动化掉，
你用 labelImg 打开时只需要 ①微调人员框 ②补画 helmet ③补画 vest。

用法：
    # 预标注一个采集目录（会在图片同目录生成 .txt）
    python tools/prelabel.py --source datasets/raw/site_a_1023

    # 覆盖已有标注（默认跳过已有 .txt，避免覆盖你手标的结果）
    python tools/prelabel.py --source datasets/raw/site_a_1023 --overwrite

    # 换权重 / 调阈值（漏检多就降 conf，误检多就升 conf）
    python tools/prelabel.py --source datasets/raw/site_a_1023 --conf 0.25

之后用 labelImg 打开同一目录，格式选 YOLO，就能看到已经框好的人，接着补框即可。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ultralytics import YOLO  # noqa: E402

from config import DEVICE, PRETRAINED_WEIGHTS, RAW_DIR  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PERSON_CLASS_ID = 0  # 本项目类别 0 = person
COCO_PERSON_ID = 0  # COCO 里 person 也是 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="用 COCO 预训练权重预标 person 框")
    p.add_argument("--source", type=Path, default=None, help="图片目录（默认 datasets/raw 下全部子目录）")
    p.add_argument("--weights", type=Path, default=PRETRAINED_WEIGHTS, help="COCO 预训练权重")
    p.add_argument("--conf", type=float, default=0.35, help="置信度阈值，漏检多就降到 0.2，误检多就升到 0.45")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--overwrite", action="store_true", help="覆盖已存在的 .txt（默认跳过）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise SystemExit(f"权重不存在: {args.weights}")

    roots = [args.source] if args.source else [d for d in RAW_DIR.iterdir() if d.is_dir()]
    images = []
    for r in roots:
        if not r.exists():
            raise SystemExit(f"目录不存在: {r}")
        images += [p for p in r.rglob("*") if p.suffix.lower() in IMG_EXTS]
    if not images:
        raise SystemExit("没有找到图片，请先运行 src/collect_images.py 采集。")

    model = YOLO(str(args.weights))
    written = skipped = total_boxes = empty = 0

    print(f"权重: {args.weights}")
    print(f"待处理: {len(images)} 张\n")

    for i, img in enumerate(images, 1):
        txt = img.with_suffix(".txt")
        if txt.exists() and not args.overwrite:
            skipped += 1
            continue

        # classes=[0] 让模型只检测 person，避免把其他 COCO 类也框出来
        res = model.predict(
            source=str(img), conf=args.conf, imgsz=args.imgsz, device=args.device, classes=[COCO_PERSON_ID], verbose=False
        )[0]

        lines = []
        if res.boxes is not None and len(res.boxes):
            h, w = res.orig_shape[:2]
            for box in res.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                # 裁到 0~1，防止框越过画面边缘导致标签非法
                cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
                bw, bh = min(bw, 1.0), min(bh, 1.0)
                lines.append(f"{PERSON_CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        txt.write_text("\n".join(lines), encoding="utf-8")
        written += 1
        total_boxes += len(lines)
        if not lines:
            empty += 1

        if i % 50 == 0 or i == len(images):
            print(f"  进度 {i}/{len(images)}  已写 {written} 个标签文件  共 {total_boxes} 个人员框")

    print("\n" + "=" * 56)
    print(f"生成标签文件 : {written}")
    print(f"跳过(已存在) : {skipped}")
    print(f"人员框总数   : {total_boxes}")
    print(f"未检出人员的图: {empty}  ← 这些可能是没人/远景/被遮挡，需要人工判断")
    print("=" * 56)
    print("\n下一步：用 labelImg 打开同一目录（格式选 YOLO），")
    print("        ① 微调已经画好的人员框  ② 补画 helmet（只标戴在头上的）  ③ 补画 vest")
    print("        标完运行：python src/split_dataset.py")


if __name__ == "__main__":
    main()
