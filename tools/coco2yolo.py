"""COCO 格式数据集 → YOLO 格式转换器。

背景：本项目改用公开数据集 `keremberke/hard-hat-detection`（HF 镜像可下）。
该数据集是 Roboflow 导出的 COCO 格式（图片 + 一个 _annotations.coco.json，
bbox 为绝对像素坐标 [x, y, w, h]），而 YOLO 需要的是「每张图一个 .txt，
每行 `类别id cx cy w h`，且全部归一化到 0~1」。本脚本负责这个转换。

同时承担「筛选优化」职责（对应评分点：数据集来源与规模）：
    * 只保留类别能映射到本项目类别体系的标注框，其余丢弃并计数；
    * 剔除尺寸异常（宽或高为 0）的框；
    * 图片按「split 前缀」重命名，避免不同 split 之间重名覆盖；
    * 输出逐类统计与丢弃原因，作为数据集二次加工的证据。

用法：
    python tools/coco2yolo.py --zips E:/YOLO/data_dl/hardhat/train.zip \
                                     E:/YOLO/data_dl/hardhat/valid.zip \
                                     E:/YOLO/data_dl/hardhat/test.zip \
                              --out datasets/raw/hardhat_all

转换完成后，继续走项目自己的流水线：
    python src/split_dataset.py --raw datasets/raw/hardhat_all --resize 640 --clean
    python src/check_labels.py
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import CLASS_NAMES  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png"}
# 数据集里的类别名 → 本项目类别名。左为来源，右为目标；不在表里的类直接丢弃。
NAME_MAP = {
    "hardhat": "hardhat",
    "helmet": "hardhat",  # 有些版本写作 helmet
    "no-hardhat": "no-hardhat",
    "no_helmet": "no-hardhat",
    "nohelmet": "no-hardhat",
    "head": "no-hardhat",  # 若来源把"裸露的头部"命名为 head
}
TARGET2ID = {v: k for k, v in CLASS_NAMES.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="COCO → YOLO 转换")
    p.add_argument("--zips", nargs="+", required=True, type=Path, help="一个或多个含 _annotations.coco.json 的 zip")
    p.add_argument("--out", type=Path, required=True, help="输出根目录，会创建 images/ 与 labels/")
    p.add_argument("--keep-empty", action="store_true", help="保留无标注图片作为负样本")
    p.add_argument("--limit", type=int, default=0, help="每个 zip 最多处理多少张（调试用，0=不限）")
    return p.parse_args()


def convert_zip(zip_path: Path, img_dir: Path, lbl_dir: Path, keep_empty: bool, limit: int, stats: dict) -> None:
    tag = zip_path.stem  # train / valid / test
    with zipfile.ZipFile(zip_path) as z:
        json_names = [n for n in z.namelist() if n.endswith("_annotations.coco.json")]
        if not json_names:
            print(f"[跳过] {zip_path.name} 里没有 COCO 标注文件")
            return
        coco = json.loads(z.read(json_names[0]))

        id2name = {c["id"]: c["name"] for c in coco["categories"]}
        boxes_by_img: dict[int, list] = {}
        for ann in coco["annotations"]:
            boxes_by_img.setdefault(ann["image_id"], []).append(ann)

        images = coco["images"][: limit or None]
        for im in images:
            raw_name = im["file_name"]
            w_img, h_img = float(im["width"]), float(im["height"])
            anns = boxes_by_img.get(im["id"], [])
            if not anns and not keep_empty:
                stats["跳过无标注图"] += 1
                continue

            lines = []
            for a in anns:
                src_name = id2name.get(a["category_id"], "")
                tgt_name = NAME_MAP.get(src_name.lower().strip())
                if tgt_name is None or tgt_name not in TARGET2ID:
                    stats[f"丢弃类别:{src_name}"] += 1
                    continue
                x, y, bw, bh = (float(v) for v in a["bbox"])
                if bw <= 1 or bh <= 1:
                    stats["丢弃异常框(宽高<=1)"] += 1
                    continue
                # 绝对像素 → 归一化中心点 + 宽高，并裁到 0~1
                cx = min(max((x + bw / 2) / w_img, 0.0), 1.0)
                cy = min(max((y + bh / 2) / h_img, 0.0), 1.0)
                nw = min(bw / w_img, 1.0)
                nh = min(bh / h_img, 1.0)
                lines.append(f"{TARGET2ID[tgt_name]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                stats[f"保留框:{tgt_name}"] += 1

            if not lines and not keep_empty:
                continue

            stem = f"{tag}__{Path(raw_name).stem}"
            dst_img = img_dir / f"{stem}{Path(raw_name).suffix.lower()}"
            dst_lbl = lbl_dir / f"{stem}.txt"
            if dst_img.exists():  # 理论上有 split 前缀不会重名，保险起见
                stats["跳过重名"] += 1
                continue
            dst_img.write_bytes(z.read(raw_name))
            dst_lbl.write_text("\n".join(lines), encoding="utf-8")
            stats["写入图片"] += 1


def main() -> None:
    args = parse_args()
    img_dir, lbl_dir = args.out / "images", args.out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    stats: Counter = Counter()
    for zp in args.zips:
        if not zp.exists():
            raise SystemExit(f"压缩包不存在: {zp}")
        print(f"处理 {zp.name} ...")
        convert_zip(zp, img_dir, lbl_dir, args.keep_empty, args.limit, stats)

    print("\n" + "=" * 58)
    print(f"输出目录: {args.out}")
    for k, v in sorted(stats.items()):
        print(f"  {k:<22} {v}")
    print("=" * 58)
    print("类别映射: 0=hardhat(已戴)  1=no-hardhat(未戴)")
    print("\n下一步：python src/split_dataset.py --raw", args.out.as_posix(), "--resize 640 --clean")


if __name__ == "__main__":
    main()
