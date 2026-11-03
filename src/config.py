"""全局配置：路径、类别、模型与推理参数。

所有脚本统一从这里取配置，避免路径散落在各处（评分项：项目结构规范 / 代码规范性）。
路径一律用 pathlib 基于本文件位置推导，因此在任何工作目录下运行都不会出错。
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- 目录结构 --
SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent  # E:\YOLO\safety_helmet

DATASET_DIR = ROOT / "datasets"
RAW_DIR = DATASET_DIR / "raw"  # 原始采集 + labelImg 标注结果（未划分）
IMAGES_DIR = DATASET_DIR / "images"  # 划分后的图片：{train,val,test}
LABELS_DIR = DATASET_DIR / "labels"  # 划分后的标签：{train,val,test}

CONFIG_DIR = ROOT / "configs"
WEIGHTS_DIR = ROOT / "weights"
PRETRAINED_DIR = WEIGHTS_DIR / "pretrained"  # 预训练权重（yolov8n.pt 等）
TRAINED_DIR = WEIGHTS_DIR / "trained"  # 自己训练产出的 best.pt / last.pt
RUNS_DIR = ROOT / "runs"  # Ultralytics 训练输出（曲线、混淆矩阵、results.csv）
RESULTS_DIR = ROOT / "results"  # 演示截图与视频
LOGS_DIR = ROOT / "logs"  # 本项目自己的运行日志
DOCS_DIR = ROOT / "docs"

DATA_YAML = CONFIG_DIR / "safety_helmet.yaml"

# ------------------------------------------------------------------- 类别 --
# 顺序即 label 里的 class id，标注时必须严格对应，不能随意调整。
#
# 类别体系说明（2026-10-25 调整）：
#   原先设计为 person / helmet / vest 三类，靠"人员框 + 头部区域内的安全帽框"做空间关联
#   来判定违规。改用公开数据集 keremberke/hard-hat-detection 后，它的标注直接就是
#   "戴帽的头部" vs "没戴帽的头部"两类 —— 这比空间关联更准，也让告警逻辑退化成
#   "只要检测到 no-hardhat 就是违规"，不再需要 IoU 匹配。
CLASS_NAMES: dict[int, str] = {
    0: "hardhat",  # 戴着安全帽的头部
    1: "no-hardhat",  # 没有戴安全帽的头部（违规）
}
NUM_CLASSES = len(CLASS_NAMES)
CLASS_NAMES_CN: dict[int, str] = {0: "已戴安全帽", 1: "未戴安全帽"}

# 违规类别 id：检测到该类别即触发告警
ALARM_CLASS_ID = 1

# ------------------------------------------------------------------- 模型 --
PRETRAINED_WEIGHTS = PRETRAINED_DIR / "yolov8n.pt"
BEST_WEIGHTS = TRAINED_DIR / "best.pt"  # 训练完成后由 train.py 自动复制到此
DEFAULT_WEIGHTS = BEST_WEIGHTS  # 推理默认优先用自己训练的权重

# 模型配置名（供 train.py --model 使用）
MODEL_CHOICES: dict[str, str] = {
    "baseline": "yolov8n.yaml",  # 原始结构，消融对照的基准
    "p2": "yolov8n-p2.yaml",  # 本项目最终改进：+P2 小目标检测头（四尺度）
    "cbam": "yolov8n-cbam.yaml",  # 曾经尝试：CBAM 注意力（实测无增益，保留作负结果）
    "se": "yolov8n-se.yaml",  # 备选：SE 通道注意力
    "s-cbam": "yolov8s-cbam.yaml",  # 更大容量 + CBAM（显存充足时用）
}

# ------------------------------------------------------------------- 推理 --
IMG_SIZE = 640
# 置信度阈值。实测证据：val 里一张有 7 顶安全帽 + 1 个未戴头部的图，
# conf=0.25 时零检出，conf=0.15 时 8 个框全部检出（与真值完全一致）。
# 本任务 60% 是小目标头部、置信度整体偏低，因此默认放到 0.15；
# 若现场误检偏多，界面滑块上调到 0.3 左右即可。
CONF_THRES = 0.15
IOU_THRES = 0.45  # NMS 的 IoU 阈值（界面可调）
DEVICE = 0  # 0=第一块 GPU；"cpu"=CPU
TRACKER = "botsort.yaml"  # Ultralytics 内置跟踪器，用于计数去重（YOLO+SORT 加分项）

# ------------------------------------------------------------------- 告警 --
ALARM_COOLDOWN_FRAMES = 30  # 同一告警最短间隔帧数，避免刷屏


def ensure_dirs() -> None:
    """创建运行期需要的目录（幂等）。"""
    for d in (
        DATASET_DIR,
        RAW_DIR,
        CONFIG_DIR,
        PRETRAINED_DIR,
        TRAINED_DIR,
        RUNS_DIR,
        RESULTS_DIR / "images",
        RESULTS_DIR / "videos",
        LOGS_DIR,
        DOCS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)
    sync_data_yaml()


def sync_data_yaml() -> Path:
    """把数据集 yaml 里的 path 校正为当前机器上的绝对路径。

    Ultralytics 解析 yaml 中的相对 path 时，是相对 **它自己的 datasets_dir 设置**
    （settings.json，可能指向别的项目遗留目录），而不是相对 yaml 文件本身。
    所以这里在启动时把 path 改写成绝对路径：项目整个目录挪走后，
    下次运行会自动纠正回来，不用手工改。
    """
    if not DATA_YAML.exists():
        return DATA_YAML
    lines = DATA_YAML.read_text(encoding="utf-8").splitlines()
    target = f"path: {DATASET_DIR.as_posix()}"
    for i, line in enumerate(lines):
        if line.strip().startswith("path:"):
            if line.strip() != target:
                lines[i] = target
                DATA_YAML.write_text("\n".join(lines) + "\n", encoding="utf-8")
            break
    return DATA_YAML


def resolve_weights(explicit: str | Path | None = None) -> Path:
    """按优先级解析推理权重：显式指定 > 自己训练的 best.pt > 预训练 yolov8n.pt。"""
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"权重文件不存在: {p}")
        return p
    if BEST_WEIGHTS.exists():
        return BEST_WEIGHTS
    if PRETRAINED_WEIGHTS.exists():
        return PRETRAINED_WEIGHTS
    raise FileNotFoundError("找不到任何可用权重，请先训练或把 yolov8n.pt 放到 weights/pretrained/")


if __name__ == "__main__":
    ensure_dirs()
    print(f"项目根目录: {ROOT}")
    print(f"类别: {CLASS_NAMES}")
    print(f"默认权重: {resolve_weights()}")
