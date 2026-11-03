"""模型训练入口。

一行命令开始训练，并把本次实验的超参数、数据集、权重路径完整记录下来，
直接作为"超参数调优记录 / 训练过程合理性"的佐证材料。

用法：
    # 改进版（YOLOv8n + CBAM 注意力，本项目推荐配置）
    python src/train.py --model cbam --name exp_cbam

    # 基线对照（原始 YOLOv8n 结构，用于消融实验）
    python src/train.py --model baseline --name exp_base --hyp configs/default_hyp.yaml

    # 断点续训 / 换更大模型
    python src/train.py --model s-cbam --name exp_s_cbam --batch 8
    python src/train.py --resume runs/exp_cbam/weights/last.pt
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

from config import (
    DATA_YAML,
    DEVICE,
    IMG_SIZE,
    IMAGES_DIR,
    LOGS_DIR,
    MODEL_CHOICES,
    PRETRAINED_WEIGHTS,
    RUNS_DIR,
    TRAINED_DIR,
    ensure_dirs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="训练安全帽检测模型")
    p.add_argument("--model", default="cbam", help=f"模型配置: {list(MODEL_CHOICES)} 或 yaml/pt 路径")
    p.add_argument("--data", type=Path, default=DATA_YAML, help="数据集 yaml")
    p.add_argument("--hyp", type=Path, default=None, help="超参文件；缺省用 Ultralytics 默认值")
    p.add_argument("--epochs", type=int, default=None, help="训练轮数（缺省取 hyp 里的值或 100）")
    p.add_argument("--imgsz", type=int, default=IMG_SIZE)
    p.add_argument("--batch", type=int, default=None, help="批大小；8GB 显存 640 分辨率建议 16")
    p.add_argument("--device", default=DEVICE, help="0=GPU, cpu=CPU")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--name", default=None, help="实验名（决定 runs 下的目录名）")
    p.add_argument("--pretrained", action="store_true", default=True, help="从预训练权重微调（默认开）")
    p.add_argument("--amp", action="store_true", help="开启混合精度；离线环境会触发联网自检，默认关闭")
    p.add_argument("--resume", default=None, help="从 last.pt 续训")
    p.add_argument(
        "--no-default-alias",
        action="store_true",
        help="只归档到 weights/trained/<实验名>_best.pt，不覆盖默认推理权重（做对照实验时用）",
    )
    p.add_argument("--cache", action="store_true", help="把图片缓存进内存加速（内存 ≥16G 再用）")
    return p.parse_args()


def build_model(spec: str) -> YOLO:
    """按名称或路径构建模型。

    传入 'cbam' 这类别名时，先按改进结构建网，再加载 YOLOv8 预训练权重：
    Ultralytics 的 load() 会自动做层名匹配，只继承形状一致的参数
    （检测头因为类别数 3 != 80 而被跳过），因此既保留 COCO 预训练的通用特征，
    又用上了自定义的注意力结构。
    """
    yaml_name = MODEL_CHOICES.get(spec, spec)
    model = YOLO(str(yaml_name))
    if spec in MODEL_CHOICES and PRETRAINED_WEIGHTS.exists():
        model.load(str(PRETRAINED_WEIGHTS))
        print(f"[模型] 结构={yaml_name}  已加载预训练权重={PRETRAINED_WEIGHTS.name}")
    else:
        print(f"[模型] 结构={yaml_name}  未使用预训练权重")
    return model


def main() -> None:
    args = parse_args()
    ensure_dirs()

    if not args.data.exists():
        raise SystemExit(f"数据集配置不存在: {args.data}")
    train_dir = IMAGES_DIR / "train"
    if not train_dir.exists() or not any(train_dir.iterdir()):
        raise SystemExit(
            "datasets/images/train 为空。请先运行：\n"
            "  python src/collect_images.py   # 采集\n"
            "  （用 labelImg 标注）\n"
            "  python src/split_dataset.py    # 清洗划分"
        )

    exp_name = args.name or f"{args.model}_{datetime.now().strftime('%m%d_%H%M')}"

    train_kwargs: dict = {
        "data": str(args.data),
        "epochs": args.epochs or 100,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "project": str(RUNS_DIR),
        "name": exp_name,
        "exist_ok": True,
        "amp": args.amp,
        "cache": args.cache,
        "plots": True,  # 生成 PR 曲线、混淆矩阵、标签分布图（文档要用）
        "val": True,
        "seed": 42,
        # 注意：deterministic=True 会让 cuDNN 只用确定性算子，实测在本机把单步耗时从
        # 约 330ms 拖到 700ms 以上（约 2 倍）。可复现性靠 seed=42 保证已足够，
        # 因此关闭确定性模式，换取训练速度。
        "deterministic": False,
    }
    if args.batch is not None:
        train_kwargs["batch"] = args.batch
    if args.hyp is not None:
        if not args.hyp.exists():
            raise SystemExit(f"超参文件不存在: {args.hyp}")
        train_kwargs["cfg"] = str(args.hyp)

    if args.resume:
        # 断点续训：必须传 resume=True，否则 Ultralytics 只会"加载这些权重重新开一轮新训练"，
        # 而不是接着原实验的 epoch / 优化器状态 / 学习率调度继续跑。
        resume_ckpt = Path(args.resume)
        if not resume_ckpt.exists():
            raise SystemExit(f"续训权重不存在: {resume_ckpt}\n（应指向某个实验目录下的 weights/last.pt）")
        model = YOLO(str(resume_ckpt))
        train_kwargs["resume"] = True
        if model.trainer is not None:
            exp_name = Path(model.trainer.save_dir).name  # 续训沿用原实验目录
        print(f"[续训] 从 {resume_ckpt} 继续，实验目录: {exp_name}")
    else:
        model = build_model(args.model)
    results = model.train(**train_kwargs)

    # ---------------- 训练后归档 ----------------
    save_dir = Path(getattr(model.trainer, "save_dir", RUNS_DIR / exp_name))
    exp_name = save_dir.name  # 续训时以原实验目录名为准，保证日志归档到同一个实验
    best = save_dir / "weights" / "best.pt"
    if best.exists():
        TRAINED_DIR.mkdir(parents=True, exist_ok=True)
        target = TRAINED_DIR / f"{exp_name}_best.pt"
        shutil.copy2(best, target)
        # 对照实验（baseline / SE）不覆盖默认推理权重，否则界面加载的就不是主模型了
        if not args.no_default_alias:
            shutil.copy2(best, TRAINED_DIR / "best.pt")
            print(f"[归档] 默认推理权重已更新 → {TRAINED_DIR / 'best.pt'}")
        print(f"[归档] best.pt → {target}")
        print(f"[归档] 默认推理权重 → {TRAINED_DIR / 'best.pt'}")

    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "exp_name": exp_name,
        "model_spec": args.model,
        "data": str(args.data),
        "hyp": str(args.hyp) if args.hyp else "ultralytics default",
        "epochs": train_kwargs["epochs"],
        "imgsz": args.imgsz,
        "batch": train_kwargs.get("batch", "auto"),
        "device": args.device,
        "amp": args.amp,
        "save_dir": str(save_dir),
        "metrics": {k: float(v) for k, v in getattr(results, "results_dict", {}).items()},
    }
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    rec_path = LOGS_DIR / f"train_{exp_name}.json"
    rec_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[记录] 本次实验配置与指标 → {rec_path}")
    print(f"[曲线] {save_dir / 'results.png'}   PR 曲线: {save_dir / 'BoxPR_curve.png'}")
    print(f"\n下一步：python src/val.py --weights {TRAINED_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()
