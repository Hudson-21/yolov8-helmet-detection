"""标注质量复核 —— 训练前的最后一道检查（评分项：数据标注规范性 6 分）。

检查内容：
    1. 图片与标签是否一一对应（图片无标签 / 标签无图片）；
    2. 每行是否为 5 个字段、class id 是否在 [0, nc) 内；
    3. 归一化坐标是否都在 0~1、宽高是否 > 0（labelImg 偶尔会写出越界框）；
    4. 空标签（整张图没有目标）统计；
    5. **跨集合重复检测**：train 与 val/test 是否存在完全相同的图片（数据泄漏），
       这是答辩时经常被追问的点，提前自查干净；
    6. 目标尺寸分布：统计框宽高占比，判断是否属于小目标场景、imgsz 该怎么选。

用法：
    python src/check_labels.py
    python src/check_labels.py --strict     # 发现任何问题即以非 0 退出码结束（可用于流水线）
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from config import CLASS_NAMES, CLASS_NAMES_CN, IMAGES_DIR, LABELS_DIR, NUM_CLASSES

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="复核 YOLO 标注质量")
    p.add_argument("--strict", action="store_true", help="发现问题时返回非 0 退出码")
    return p.parse_args()


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    problems: list[str] = []

    print("=" * 70)
    print("标注复核报告")
    print("=" * 70)

    hashes: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    size_buckets = Counter()
    per_class_images: Counter = Counter()
    grand_boxes = 0

    for split in SPLITS:
        img_dir, lbl_dir = IMAGES_DIR / split, LABELS_DIR / split
        images = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in IMG_EXTS]) if img_dir.exists() else []
        labels = sorted(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

        img_stems = {p.stem for p in images}
        lbl_stems = {p.stem for p in labels}
        orphan_labels = lbl_stems - img_stems
        no_label_imgs = img_stems - lbl_stems

        boxes, empty, bad = 0, 0, 0
        cls_counter: Counter = Counter()

        for lbl in labels:
            lines = [ln.strip() for ln in lbl.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
            if not lines:
                empty += 1
                continue
            file_bad = False
            for ln in lines:
                parts = ln.split()
                if len(parts) != 5:
                    problems.append(f"[{split}] {lbl.name}: 字段数 {len(parts)} != 5 → {ln[:40]}")
                    file_bad = True
                    continue
                try:
                    cid = int(float(parts[0]))
                    cx, cy, w, h = (float(v) for v in parts[1:])
                except ValueError:
                    problems.append(f"[{split}] {lbl.name}: 存在非数值字段 → {ln[:40]}")
                    file_bad = True
                    continue
                if not 0 <= cid < NUM_CLASSES:
                    problems.append(f"[{split}] {lbl.name}: class id {cid} 超出 0~{NUM_CLASSES - 1}")
                    file_bad = True
                    continue
                if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
                    problems.append(f"[{split}] {lbl.name}: 坐标越界 {cx:.3f},{cy:.3f},{w:.3f},{h:.3f}")
                    file_bad = True
                if w <= 0 or h <= 0:
                    problems.append(f"[{split}] {lbl.name}: 宽高为 0")
                    file_bad = True
                boxes += 1
                cls_counter[CLASS_NAMES[cid]] += 1
                # 按 COCO 口径：面积占比 < 0.01 视为小目标
                area = w * h
                size_buckets["小目标(<1%)" if area < 0.01 else ("中目标(1%~10%)" if area < 0.1 else "大目标(>10%)")] += 1
            if file_bad:
                bad += 1

        for stem in img_stems:
            per_class_images[stem] = 1
        for img in images:
            hashes[md5_of(img)][split].append(img.name)

        grand_boxes += boxes
        print(f"\n[{split}]")
        print(f"  图片: {len(images):4d}   标签: {len(labels):4d}   标注框: {boxes:5d}")
        print(f"  空标签: {empty}   含错误标签: {bad}")
        if orphan_labels:
            problems.append(f"[{split}] 有 {len(orphan_labels)} 个标签找不到同名图片，例如 {sorted(orphan_labels)[:3]}")
        if no_label_imgs:
            problems.append(f"[{split}] 有 {len(no_label_imgs)} 张图片没有标签（可能漏标），例如 {sorted(no_label_imgs)[:3]}")
        if cls_counter:
            detail = "，".join(f"{k}={v}" for k, v in cls_counter.most_common())
            print(f"  类别分布: {detail}")

    # ---------------- 跨集合泄漏 ----------------
    leaks = [(h, d) for h, d in hashes.items() if len(d) > 1]
    if leaks:
        for _, d in leaks[:10]:
            where = " / ".join(f"{k}:{v[0]}" for k, v in d.items())
            problems.append(f"重复图片跨集合出现（数据泄漏）→ {where}")
        if len(leaks) > 10:
            problems.append(f"... 另有 {len(leaks) - 10} 组跨集合重复图片")

    # ---------------- 汇总 ----------------
    print("\n" + "-" * 70)
    print(f"总标注框: {grand_boxes}")
    if size_buckets:
        total = sum(size_buckets.values())
        print("目标尺寸分布: " + "，".join(f"{k}={v}({v / total:.1%})" for k, v in size_buckets.items()))
        small = size_buckets["小目标(<1%)"] / max(total, 1)
        if small > 0.4:
            print("  ⚠ 小目标占比高，建议 imgsz ≥ 960 训练（显存不够时退 768，batch 相应减半）")
    print(f"类别中文名: {', '.join(f'{k}={v}' for k, v in CLASS_NAMES_CN.items())}")

    if problems:
        print(f"\n发现 {len(problems)} 个问题：")
        for p in problems[:40]:
            print("  ✗", p)
        if len(problems) > 40:
            print(f"  ... 其余 {len(problems) - 40} 条略")
        print("\n建议：用 labelImg 打开对应图片修正后重新运行 split_dataset.py 与 check_labels.py")
        if args.strict:
            raise SystemExit(1)
    else:
        print("\n✓ 未发现标注问题，可以开始训练")
    print("=" * 70)


if __name__ == "__main__":
    main()
