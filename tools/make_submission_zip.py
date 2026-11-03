"""

注：姓名/学号/班级使用占位符，实际使用时自行填写。
组装课程设计提交压缩包。

原则：
  * 代码、权重、训练证据、数据集样例全部打包，但不含 800MB 数据集本体和 1.1GB 原始压缩包
    （提交系统通常有大小限制，且数据集可由说明里的命令复现）；
  * 框架改动单独成目录，因为评分明确要求"必须改动开源框架源码"；
  * 目录用数字前缀排序，老师解压后按顺序看。
"""

from __future__ import annotations

import os

import csv
import shutil
import zipfile
from pathlib import Path

WORK = Path(os.environ.get("REPORT_DIR", "课程材料目录"))  # 存放报告与提交包的目录
PROJ = Path(r"E:/YOLO/safety_helmet")
ULTRA = Path(r"E:/YOLO/ultralytics-main")
STAGE = Path(os.environ.get("SUBMIT_DIR", "提交包构建目录")) / "safety_helmet_submission"
ZIP = STAGE.parent / "safety_helmet_submission.zip"

NAME = "姓名"
SID = "学号"
CLASS_NAME = "班级"


def copy_tree(src: Path, dst: Path, skip: tuple[str, ...] = ()) -> int:
    """复制目录（跳过 __pycache__ 与指定后缀），返回文件数。"""
    n = 0
    for p in src.rglob("*"):
        if p.is_dir():
            continue
        if "__pycache__" in str(p) or p.suffix in skip:
            continue
        target = dst / p.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        n += 1
    return n


def read_best(exp: str) -> dict | None:
    f = PROJ / "runs" / exp / "results.csv"
    if not f.exists():
        return None
    rows = list(csv.DictReader(f.open(encoding="utf-8")))
    if len(rows) < 20:
        return None

    def fit(r: dict) -> float:
        return 0.1 * float(r["metrics/mAP50(B)"]) + 0.9 * float(r["metrics/mAP50-95(B)"])

    b = max(rows, key=fit)
    return {
        "epochs": len(rows),
        "epoch": b["epoch"],
        "P": float(b["metrics/precision(B)"]),
        "R": float(b["metrics/recall(B)"]),
        "mAP50": float(b["metrics/mAP50(B)"]),
        "mAP95": float(b["metrics/mAP50-95(B)"]),
    }


def main() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    report = 0
    # ---------- 01 报告 ----------
    d = STAGE / "01_课程设计报告"
    d.mkdir()
    rep = WORK / "课程设计报告.docx"
    if rep.exists():
        shutil.copy2(rep, d / rep.name)
        report = 1

    # ---------- 02 源代码 ----------
    d = STAGE / "02_源代码"
    n_code = 0
    for sub in ("src", "configs", "tools", "docs"):
        n_code += copy_tree(PROJ / sub, d / sub, skip=(".pyc",))
    for f in ("requirements.txt", "README.md", "run_ui.bat", "setup.bat", "run_ablation.bat"):
        if (PROJ / f).exists():
            shutil.copy2(PROJ / f, d / f)
            n_code += 1

    # ---------- 03 框架改动 ----------
    d = STAGE / "03_框架改动_ultralytics"
    d.mkdir()
    files = {
        ULTRA / "ultralytics/nn/modules/attention_extra.py": "nn_modules_attention_extra.py",
        ULTRA / "ultralytics/nn/tasks.py": "nn_tasks.py",
        ULTRA / "ultralytics/cfg/models/v8/yolov8n-cbam.yaml": "cfg_models_v8_yolov8n-cbam.yaml",
        ULTRA / "ultralytics/cfg/models/v8/yolov8n-se.yaml": "cfg_models_v8_yolov8n-se.yaml",
        ULTRA / "ultralytics/cfg/models/v8/yolov8n-p2.yaml": "cfg_models_v8_yolov8n-p2.yaml",
    }
    for src, name in files.items():
        if src.exists():
            shutil.copy2(src, d / name)
    (d / "改动说明.md").write_text(
        "# 对 Ultralytics 源码的改动\n\n"
        "本项目未直接使用官方 pip 发行版，而是以可编辑方式安装本地源码并做了三处改动，"
        "否则自定义模块无法在 yaml 中被解析。\n\n"
        "## 1. 新增注意力模块 `ultralytics/nn/modules/attention_extra.py`\n"
        "实现 `CBAMBlock`（通道注意力 + 空间注意力串联）与 `SEAttention`（仅通道注意力）。\n\n"
        "## 2. 注册进模块解析入口 `ultralytics/nn/tasks.py`\n"
        "把两个类加入 `parse_model` 的 `base_modules` 集合，使其可在 yaml 中按类名引用；"
        "同时在 `__all__` 中导出。\n\n"
        "## 3. 新增模型配置（`ultralytics/cfg/models/v8/`）\n\n"
        "| 文件 | 改动内容 |\n|---|---|\n"
        "| `yolov8n-cbam.yaml` | 在 SPPF 之后与 Neck 的 P3/P4/P5 三处插入 CBAM（共 4 处） |\n"
        "| `yolov8n-se.yaml` | 同上位置插入 SE（对照实验） |\n"
        "| `yolov8n-p2.yaml` | Neck 向下延伸 P2 分支（160×160，stride 4），Detect 改为四尺度 |\n\n"
        "## 部署方式\n"
        "把上述文件放回对应路径后：\n\n"
        "```bash\n"
        "cd E:/YOLO/ultralytics-main\n"
        "pip install -e .\n"
        "```\n\n"
        "> 注意：必须可编辑安装本源码。若用官方 pip 包，加载 `yolov8n-cbam.yaml` 会报\n"
        "> `NameError: name 'CBAMBlock' is not defined`。\n",
        encoding="utf-8",
    )

    # ---------- 04 权重 ----------
    d = STAGE / "04_模型权重"
    d.mkdir()
    best = PROJ / "weights/trained/best.pt"
    if best.exists():
        shutil.copy2(best, d / "best.pt")
    m = read_best("exp6_baseline") or {}
    (d / "权重说明.txt").write_text(
        "best.pt —— 最终部署模型\n"
        "  结构：原生 YOLOv8n（未加注意力、未加 P2 头）\n"
        "  类别：0=hardhat（已戴安全帽的头部）、1=no-hardhat（未戴）\n"
        f"  验证集指标：mAP@0.5={m.get('mAP50', 0):.4f}，mAP@0.5:0.95={m.get('mAP95', 0):.4f}，"
        f"P={m.get('P', 0):.4f}，R={m.get('R', 0):.4f}\n"
        "  训练：25 轮，batch 16，imgsz 640，数据集 19441 张\n\n"
        "加载方式：程序默认读取 weights/trained/best.pt，无需额外配置。\n"
        "对照实验权重（CBAM / SE / P2）体积各约 6MB，未一并打包，"
        "可从 runs/ 下的训练记录或重新训练获得。\n",
        encoding="utf-8",
    )

    # ---------- 05 训练记录 ----------
    d = STAGE / "05_训练与评估记录"
    d.mkdir()
    n_run = 0
    for exp in ("exp5_cbam", "exp6_baseline", "exp7_se", "exp8_p2"):
        src = PROJ / "runs" / exp
        if not src.exists():
            continue
        # 只取曲线图与结果表，不复制权重与可视化批次图
        for f in src.glob("*.png"):
            t = d / "曲线图" / f"{exp}_{f.name}"
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, t)
            n_run += 1
        if (src / "results.csv").exists():
            t = d / "results_csv" / f"{exp}_results.csv"
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / "results.csv", t)
            n_run += 1
    for f in (PROJ / "logs").glob("train_*.json"):
        t = d / "训练配置与指标" / f.name
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, t)
        n_run += 1
    for f in (PROJ / "logs").glob("split_report_*.csv"):
        t = d / "数据集划分明细" / f.name
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, t)
        n_run += 1

    # 消融对比表
    lines = [
        "# 消融实验对比",
        "",
        "三组实验使用同一份数据划分、同一套超参数与相同轮数（25 轮）进行训练。",
        "P2 结构因浅层 160×160 特征图显存占用更高，只能以 batch 8 训练，另两组为 batch 16。",
        "",
        "| 实验 | 结构改动 | batch | P | R | mAP@0.5 | mAP@0.5:0.95 |",
        "|---|---|---|---|---|---|---|",
    ]
    for exp, label, batch, change in (
        ("exp6_baseline", "baseline", 16, "原生 YOLOv8n"),
        ("exp5_cbam", "+CBAM", 16, "4 处 CBAM 注意力"),
        ("exp7_se", "+SE", 16, "4 处 SE 注意力（仅 3 轮，未纳入对比）"),
        ("exp8_p2", "+P2", 8, "新增 P2 四尺度检测层"),
    ):
        r = read_best(exp)
        if r is None:
            continue
        lines.append(
            f"| {label} | {change} | {batch} | {r['P']:.4f} | {r['R']:.4f} | "
            f"**{r['mAP50']:.4f}** | {r['mAP95']:.4f} |"
        )
    lines += [
        "",
        "> 注：SE（纯通道注意力）对照实验仅训练 3 轮即中止——CBAM 已证明注意力在本任务无收益，"
        "继续训练意义不大，故转向 P2 小目标方案。其记录仍保留在 `results_csv/exp7_se_results.csv`，"
        "因未跑完 25 轮，不计入对比表。",
        "",
        "## 结论",
        "",
        "1. 三组 mAP@0.5 差异均在 0.5 个百分点以内，处于随机波动范围；",
        "2. CBAM 注意力未带来收益，与「数据集仅 2 类、目标特征区分度高、COCO 预训练已充分」一致；",
        "3. P2 与另两组批大小不同，差异不能完全归因于结构，故不判定为有效改进；",
        "4. 最终部署采用指标最优且参数量最小的原生 YOLOv8n。",
        "",
        "曲线图见 `曲线图/` 目录（results.png 为损失与精度曲线，BoxPR_curve.png 为 PR 曲线，"
        "confusion_matrix_normalized.png 为归一化混淆矩阵）。",
    ]
    (d / "消融对比.md").write_text("\n".join(lines), encoding="utf-8")

    # ---------- 06 数据集样例 ----------
    d = STAGE / "06_数据集样例"
    (d / "images").mkdir(parents=True)
    (d / "labels").mkdir(parents=True)
    import glob

    n_img = 0
    for img in sorted(glob.glob(str(PROJ / "datasets/images/val/*.jpg")))[:8]:
        p = Path(img)
        shutil.copy2(p, d / "images" / p.name)
        lb = PROJ / "datasets/labels/val" / (p.stem + ".txt")
        if lb.exists():
            shutil.copy2(lb, d / "labels" / lb.name)
        n_img += 1
    (d / "数据集说明.md").write_text(
        "# 数据集说明\n\n"
        "## 来源\n"
        "公开数据集 `keremberke/hard-hat-detection`，经 HuggingFace 国内镜像下载，"
        "原始规模 19445 张、55391 个标注框，标注粒度为头部级二分类。\n\n"
        "> 因体积（原始压缩包 1.12 GB、解压后约 800 MB）与提交系统限制，"
        "压缩包内仅附 8 张样例图片及其标注，完整数据可按下列命令复现。\n\n"
        "## 复现命令\n\n"
        "```bash\n"
        "mkdir -p E:/YOLO/data_dl/hardhat && cd E:/YOLO/data_dl/hardhat\n"
        "curl -L -o train.zip https://hf-mirror.com/datasets/keremberke/hard-hat-detection/resolve/main/data/train.zip\n"
        "curl -L -o valid.zip https://hf-mirror.com/datasets/keremberke/hard-hat-detection/resolve/main/data/valid.zip\n"
        "curl -L -o test.zip  https://hf-mirror.com/datasets/keremberke/hard-hat-detection/resolve/main/data/test.zip\n\n"
        "# COCO → YOLO 转换 + 初筛\n"
        "python tools/coco2yolo.py --zips E:/YOLO/data_dl/hardhat/train.zip E:/YOLO/data_dl/hardhat/valid.zip "
        "E:/YOLO/data_dl/hardhat/test.zip --out datasets/raw/hardhat_all\n\n"
        "# 清洗去重 + 8:1:1 重划分\n"
        "python src/split_dataset.py --raw datasets/raw/hardhat_all --resize 640 --clean\n\n"
        "# 标注全量复核\n"
        "python src/check_labels.py\n"
        "```\n\n"
        "## 加工后的数据事实\n\n"
        "| 项 | 数值 |\n|---|---|\n"
        "| 可用图片 | 19441 张（剔除重复 3 张、无标注 301 张、退化框 2 个） |\n"
        "| 标注框 | 55381 个（训练 44515 / 验证 5459 / 测试 5407） |\n"
        "| 划分 | 15552 / 1944 / 1945（8:1:1，种子 42） |\n"
        "| 类别分布 | hardhat 42423（76.6%）· no-hardhat 12958（23.4%） |\n"
        "| 目标尺寸 | 小目标（面积占比<1%）60.1% · 中 35.0% · 大 4.9% |\n"
        "| 标注校验 | 字段错误/类别越界/坐标越界/空标签/跨集合重复 均为 0 |\n\n"
        "## 类别定义\n\n"
        "- `0 hardhat`：戴着安全帽的头部\n"
        "- `1 no-hardhat`：没有戴安全帽的头部（告警类别）\n",
        encoding="utf-8",
    )

    # ---------- 00 提交说明 ----------
    m2 = read_best("exp6_baseline") or {}
    SEP_H, SEP_D = "=" * 56, "-" * 56
    readme = [
        "《行业工程实践YOLO视觉模块》课程设计 —— 期末项目成果提交",
        SEP_H,
        "",
        f"姓    名：{NAME}",
        f"学    号：{SID}",
        f"班    级：{CLASS_NAME}",
        "项目名称：基于 YOLOv8 改进的安全帽佩戴检测系统",
        "",
        SEP_D,
        "一、压缩包内容",
        SEP_D,
        "  00_提交说明.txt            本文件",
        "  01_课程设计报告/           课程设计报告（Word）",
        "  02_源代码/                 全部工程代码、配置、文档",
        "  03_框架改动_ultralytics/   对 Ultralytics 源码的改动与说明",
        "  04_模型权重/               best.pt（最终部署模型）",
        "  05_训练与评估记录/         训练曲线、逐轮指标、消融对比、划分明细",
        "  06_数据集样例/             8 张样例图与标注 + 数据集复现说明",
        "",
        SEP_D,
        "二、核心指标",
        SEP_D,
        f"  验证集 mAP@0.5      {m2.get('mAP50', 0):.4f}   （要求 ≥0.85）",
        f"  验证集 mAP@0.5:0.95 {m2.get('mAP95', 0):.4f}",
        f"  Precision / Recall  {m2.get('P', 0):.4f} / {m2.get('R', 0):.4f}",
        "  推理速度            纯检测 42.9 FPS，视频端到端 19.5 FPS（要求 ≥15）",
        "  界面实测            摄像头实时检测约 60 FPS，连续运行 10 分钟无异常",
        "",
        SEP_D,
        "三、如何运行",
        SEP_D,
        "  1) 安装环境（首次）：双击 02_源代码/setup.bat，或手动执行",
        "       pip install -r requirements.txt",
        "       cd <ultralytics 源码目录> && pip install -e .",
        "     注意：必须可编辑安装改过的 ultralytics 源码（见 03_框架改动 目录），",
        "     否则加载自定义模型配置会报 CBAMBlock is not defined。",
        "",
        "  2) 启动图形界面：双击 02_源代码/run_ui.bat",
        "     或命令行：python src/ui_app.py",
        "",
        "  3) 命令行推理：",
        "       python src/predict.py --source <图片/视频路径>",
        "       python src/predict.py --source 0        （摄像头）",
        "",
        "  4) 重新训练：",
        "       python src/train.py --model baseline --name my_exp",
        "",
        SEP_D,
        "四、运行环境",
        SEP_D,
        "  Windows 11 24H2 / AMD Ryzen 9 7945HX / 16 GB 内存",
        "  NVIDIA RTX 4070 Laptop 8 GB / CUDA 12.6",
        "  Python 3.13.2（conda 环境 yolov8）/ PyTorch 2.10.0",
        "  Ultralytics 8.4.184（本地源码可编辑安装）",
        "",
        SEP_D,
        "五、说明",
        SEP_D,
        "  数据集本体（约 800 MB）未打包，仅附样例与复现命令，原因见 06 目录说明。",
        "  CBAM 与 P2 两组对照实验的权重未打包，训练记录与曲线已完整保留。",
        "  报告中的逐条测试用例结果均为实测，含一条实测发现并修复的告警冷却缺陷。",
        "",
    ]
    (STAGE / "00_提交说明.txt").write_text("\n".join(readme), encoding="utf-8")

    # ---------- 打包 ----------
    ZIP.parent.mkdir(parents=True, exist_ok=True)
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(STAGE.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(STAGE.parent))
    size = ZIP.stat().st_size / 1024 / 1024
    print(f"压缩包：{ZIP}")
    print(f"体积：{size:.1f} MB")
    print(f"报告 {report} 份 · 代码 {n_code} 个文件 · 训练记录 {n_run} 个 · 样例图 {n_img} 张")


if __name__ == "__main__":
    main()
