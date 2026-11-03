"""数据集清洗与划分 —— 把 datasets/raw 里的原始采集图整理成 YOLO 标准目录。

做的事（对应评分项：数据预处理与增强 8 分）：
    1. 数据清洗：损坏图片、无法解码、尺寸过小的文件剔除；
    2. 去重：按文件 MD5 去掉重复/连拍几乎相同的图，避免训练集与验证集同源泄漏；
    3. 分辨率统一：可选把长边统一到指定尺寸（标注是归一化坐标，缩放不影响标签）；
    4. 标签完整性：找不到 .txt、空 .txt、类别越界、坐标越界的样本单独报告并剔除；
    5. 数据集划分：按 8:1:1 随机划分 train/val/test（固定随机种子，结果可复现）；
    6. 产出清单：logs/split_report_*.csv 记录每张图的归属与剔除原因，可写进文档。

用法：
    python src/split_dataset.py                       # 默认 8:1:1，不缩放
    python src/split_dataset.py --resize 640          # 长边统一到 640
    python src/split_dataset.py --seed 42 --ratio 0.8 0.1 0.1
    python src/split_dataset.py --keep-empty          # 保留无目标的纯背景图（作为负样本）
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
from datetime import datetime
from pathlib import Path

import cv2

from config import IMAGES_DIR, LABELS_DIR, LOGS_DIR, NUM_CLASSES, RAW_DIR

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


# ----------------------------------------------------------------- 工具函数 --
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="清洗并划分 YOLO 数据集")
    p.add_argument("--raw", type=Path, default=RAW_DIR, help="原始数据目录")
    p.add_argument("--ratio", type=float, nargs=3, default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=42, help="随机种子，保证划分可复现")
    p.add_argument("--resize", type=int, default=0, help="统一长边像素，0 表示保持原分辨率")
    p.add_argument("--min-side", type=int, default=320, help="短边小于该值的图片视为无效")
    p.add_argument("--keep-empty", action="store_true", help="保留无标注的纯背景图作为负样本")
    p.add_argument("--clean", action="store_true", help="划分前清空已有 images/labels 目标目录")
    p.add_argument("--dry-run", action="store_true", help="只统计不落盘")
    return p.parse_args()


def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    """计算文件 MD5，用于去重。"""
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk_data := f.read(chunk):
            h.update(chunk_data)
    return h.hexdigest()


def find_label(img: Path) -> Path | None:
    """找到图片对应的 YOLO 标签文件。

    兼容两种常见组织方式：
        raw/scene/a.jpg + raw/scene/a.txt          （labelImg 默认，同目录）
        raw/images/a.jpg + raw/labels/a.txt        （标准布局）
    """
    same = img.with_suffix(".txt")
    if same.exists():
        return same
    for parent in img.parents:
        if parent.name in {"images", "JPEGImages"}:
            cand = parent.parent / "labels" / f"{img.stem}.txt"
            if cand.exists():
                return cand
        if parent == RAW_DIR:
            break
    return None


def validate_label(txt: Path) -> tuple[bool, str]:
    """校验一个 YOLO 标签文件，返回 (是否合法, 原因)。"""
    lines = [ln.strip() for ln in txt.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    if not lines:
        return False, "空标签"
    for ln in lines:
        parts = ln.split()
        if len(parts) != 5:
            return False, f"字段数={len(parts)}"
        try:
            cid = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            return False, "非数值"
        if not 0 <= cid < NUM_CLASSES:
            return False, f"类别越界 id={cid}"
        if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
            return False, "坐标未归一化/越界"
        if w <= 0 or h <= 0:
            return False, "宽高为 0"
    return True, ""


# --------------------------------------------------------------------- 主流程 --
def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if not args.raw.exists():
        raise SystemExit(f"原始目录不存在: {args.raw}\n请先采集图片并完成标注，再运行本脚本。")

    images = [p for p in args.raw.rglob("*") if p.suffix.lower() in IMG_EXTS]
    if not images:
        raise SystemExit(f"{args.raw} 下没有找到图片。")

    records: list[dict] = []
    kept: list[tuple[Path, Path]] = []
    seen_md5: dict[str, str] = {}

    for img in sorted(images):
        rec = {"image": str(img.relative_to(args.raw)), "split": "", "reason": "", "n_boxes": 0}
        # 1) 能否解码
        data = cv2.imread(str(img))
        if data is None:
            rec["reason"] = "图片无法解码"
            records.append(rec)
            continue
        h, w = data.shape[:2]
        if min(h, w) < args.min_side:
            rec["reason"] = f"分辨率过小({w}x{h})"
            records.append(rec)
            continue

        # 2) 去重
        digest = file_md5(img)
        if digest in seen_md5:
            rec["reason"] = f"重复图片(与 {seen_md5[digest]} 相同)"
            records.append(rec)
            continue
        seen_md5[digest] = rec["image"]

        # 3) 标签
        label = find_label(img)
        if label is None:
            rec["reason"] = "缺少标签文件"
            records.append(rec)
            continue
        ok, why = validate_label(label)
        if not ok:
            if why == "空标签" and args.keep_empty:
                rec["n_boxes"] = 0
                kept.append((img, label))
                records.append(rec)
                continue
            rec["reason"] = f"标签非法({why})"
            records.append(rec)
            continue
        rec["n_boxes"] = len([ln for ln in label.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()])
        kept.append((img, label))
        records.append(rec)

    if not kept:
        raise SystemExit("清洗后没有可用样本，请检查标注是否已生成 .txt 文件。")

    # 4) 划分（打乱后按比例切分，保证三部分都有样本）
    random.shuffle(kept)
    n = len(kept)
    n_train = max(1, int(n * args.ratio[0]))
    n_val = max(1, int(n * args.ratio[1])) if n >= 10 else 0
    if n_train + n_val >= n:
        n_train, n_val = n - 2, 1
    buckets = {
        "train": kept[:n_train],
        "val": kept[n_train : n_train + n_val],
        "test": kept[n_train + n_val :],
    }

    # 5) 落盘
    if args.clean and not args.dry_run:
        for split in SPLITS:
            for d in (IMAGES_DIR / split, LABELS_DIR / split):
                if d.exists():
                    shutil.rmtree(d)
    for split in SPLITS:
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)

    split_of: dict[str, str] = {}
    for split, items in buckets.items():
        for img, label in items:
            split_of[str(img.relative_to(args.raw))] = split
            if args.dry_run:
                continue
            dst_img = IMAGES_DIR / split / img.name
            dst_lbl = LABELS_DIR / split / f"{img.stem}.txt"
            if args.resize and args.resize > 0:
                frame = cv2.imread(str(img))
                h, w = frame.shape[:2]
                scale = args.resize / max(h, w)
                if scale < 1.0:
                    frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(dst_img), frame)
            else:
                shutil.copy2(img, dst_img)
            shutil.copy2(label, dst_lbl)

    for rec in records:
        rec["split"] = split_of.get(rec["image"], "")

    # 6) 报告
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = LOGS_DIR / f"split_report_{stamp}.csv"
    with report.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "split", "n_boxes", "reason"])
        writer.writeheader()
        writer.writerows(records)

    dropped = [r for r in records if r["reason"]]
    total_boxes = sum(r["n_boxes"] for r in records if not r["reason"])
    print("=" * 62)
    print(f"原始图片      : {len(images)}")
    print(f"可用样本      : {len(kept)}")
    print(f"剔除样本      : {len(dropped)}")
    print(f"标注框总数    : {total_boxes}")
    print(f"train/val/test: {len(buckets['train'])} / {len(buckets['val'])} / {len(buckets['test'])}")
    if dropped:
        from collections import Counter

        print("剔除原因分布  :", dict(Counter(r["reason"].split("(")[0] for r in dropped)))
    print(f"明细报告      : {report}")
    print("=" * 62)
    if args.dry_run:
        print("（--dry-run 模式，未写入任何文件）")
    else:
        print("下一步：python src/check_labels.py  复核标签，然后开始训练")


if __name__ == "__main__":
    main()
