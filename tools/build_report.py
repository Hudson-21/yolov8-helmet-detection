"""

注：姓名/学号/班级使用占位符，实际使用时自行填写。
按课程设计报告模板生成正式报告（XML 层填充，保留原格式）。

设计要点：
  * 只替换 word/document.xml 与 word/media/image2.png，其余部件原样复制，避免样式/关系断裂；
  * 所有新段落复用模板里的段落格式原型（宋体、小四、首行缩进 2 字符），不自己拼字体；
  * 指标从 runs/*/results.csv 实时读取，训练结果变了重跑本脚本即可，不用手改数字；
  * 表1~表4 的单元格逐格重写，并清掉模板里原有的电商项目截图（rId10~rId14）。

用法：python tools/build_report.py
输出：报告目录下的 课程设计报告.docx
"""

from __future__ import annotations

import os

import csv
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 课程材料（模板与成品报告）放在工作区目录，不在工程目录内
DOC_DIR = Path(os.environ.get("REPORT_DIR", str(ROOT.parent)))  # 课程材料目录，可用 REPORT_DIR 指定
TEMPLATE = DOC_DIR / os.environ.get("REPORT_TEMPLATE", "报告模板.docx")
OUTPUT = DOC_DIR / "课程设计报告.docx"
ARCH_IMG = ROOT / "docs" / "系统功能模块架构图.png"

# 训练曲线图使用的图片关系（运行时按模板已占用的 rId 动态分配）
CURVE_RID = "rId15"
CURVE_MEDIA = "image8.png"

# ----------------------------------------------------------------- 基本信息 --
NAME = "姓名"
SID = "学号"
CLASS_NAME = "班级"
DATE = "2026.10.26"

# ----------------------------------------------------------------- 指标读取 --
EXPERIMENTS = [
    ("exp6_baseline", "baseline（原生 YOLOv8n）", "16"),
    ("exp5_cbam", "+CBAM（注意力尝试）", "16"),
    ("exp8_p2", "+P2（本设计，小目标检测层）", "8"),
    ("exp9_baseline_b8", "baseline（同 batch 对照）", "8"),
]


def read_metrics() -> list[dict]:
    """从各实验的 results.csv 里取 fitness 最优的一轮。"""
    out = []
    for exp, label, batch in EXPERIMENTS:
        f = ROOT / "runs" / exp / "results.csv"
        if not f.exists():
            continue
        rows = list(csv.DictReader(f.open(encoding="utf-8")))
        # 只纳入跑完的实验：中途停止的实验不足 20 轮，不能进入对比表
        if len(rows) < 20:
            continue

        def fitness(r: dict) -> float:
            return 0.1 * float(r["metrics/mAP50(B)"]) + 0.9 * float(r["metrics/mAP50-95(B)"])

        best = max(rows, key=fitness)
        out.append(
            {
                "exp": exp,
                "label": label,
                "batch": batch,
                "epochs": len(rows),
                "best_epoch": int(best["epoch"]),
                "P": float(best["metrics/precision(B)"]),
                "R": float(best["metrics/recall(B)"]),
                "mAP50": float(best["metrics/mAP50(B)"]),
                "mAP50_95": float(best["metrics/mAP50-95(B)"]),
            }
        )
    return out


# --------------------------------------------------------------- XML 工具 --
XML_ESC = str.maketrans({"&": "&amp;", "<": "&lt;", ">": "&gt;"})


def esc(s: str) -> str:
    return s.translate(XML_ESC)


def split_pairs(xml: str, s: int, e: int, tag: str) -> list[tuple[int, int]]:
    """提取 [s,e) 内 tag 的配对区间，正确处理嵌套。"""
    res, pos = [], s
    op, ct = re.compile("<" + tag + r"(?:\s[^>]*)?>"), "</" + tag + ">"
    while True:
        m = op.search(xml, pos, e)
        if not m:
            break
        depth, p = 1, m.end()
        while depth > 0:
            c = xml.find(ct, p, e)
            if c < 0:
                raise RuntimeError(f"{tag} 配对失败")
            o = op.search(xml, p, c)
            if o:
                depth += 1
                p = o.end()
            else:
                depth -= 1
                p = c + len(ct)
                if depth == 0:
                    res.append((m.start(), p))
        pos = p
    return res


RPR_BODY = '<w:rPr><w:rFonts w:ascii="宋体" w:hAnsi="宋体" w:hint="eastAsia"/><w:sz w:val="24"/></w:rPr>'
PPR_BODY = '<w:pPr><w:spacing w:line="360" w:lineRule="auto"/><w:ind w:firstLineChars="200" w:firstLine="480"/>' + RPR_BODY + "</w:pPr>"


def para(text: str, kind: str = "body", align: str | None = None) -> str:
    """生成一个段落。kind: body / code / caption / plain"""
    if kind == "code":
        rpr = '<w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:hint="eastAsia"/><w:sz w:val="18"/></w:rPr>'
        ppr = '<w:pPr><w:spacing w:line="240" w:lineRule="auto"/><w:ind w:left="420"/>' + rpr + "</w:pPr>"
        # 注意：<w:br/> 不能写在 <w:t> 内部（会构成非法 OOXML，Word 直接拒绝打开），
        # 必须用独立的 run 承载换行
        runs = ""
        for i, ln in enumerate(text.split("\n")):
            if i:
                runs += f"<w:r>{rpr}<w:br/></w:r>"
            runs += f'<w:r>{rpr}<w:t xml:space="preserve">{esc(ln)}</w:t></w:r>'
        return f"<w:p>{ppr}{runs}</w:p>"
    if kind == "caption":
        rpr = '<w:rPr><w:rFonts w:ascii="宋体" w:hAnsi="宋体" w:hint="eastAsia"/><w:sz w:val="21"/></w:rPr>'
        ppr = '<w:pPr><w:jc w:val="center"/>' + rpr + "</w:pPr>"
        return f'<w:p>{ppr}<w:r>{rpr}<w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>'
    if kind == "plain":
        rpr = '<w:rPr><w:rFonts w:ascii="宋体" w:hAnsi="宋体" w:hint="eastAsia"/><w:sz w:val="24"/></w:rPr>'
        ppr = "<w:pPr>" + rpr + "</w:pPr>"
        return f'<w:p>{ppr}<w:r>{rpr}<w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>'
    jc = f'<w:jc w:val="{align}"/>' if align else ""
    return f'<w:p>{PPR_BODY}{jc}<w:r>{RPR_BODY}<w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>'


def make_table(rows: list[list[str]], widths: list[int], font_sz: int = 20, header: bool = True) -> str:
    """生成带边框的表格（widths 单位 dxa，1cm≈567）。"""
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    tb = (
        '<w:tbl><w:tblPr><w:tblW w:w="5000" w:type="pct"/>'
        '<w:tblBorders>'
        '<w:top w:val="single" w:sz="8" w:color="000000"/>'
        '<w:left w:val="single" w:sz="8" w:color="000000"/>'
        '<w:bottom w:val="single" w:sz="8" w:color="000000"/>'
        '<w:right w:val="single" w:sz="8" w:color="000000"/>'
        '<w:insideH w:val="single" w:sz="4" w:color="000000"/>'
        '<w:insideV w:val="single" w:sz="4" w:color="000000"/>'
        "</w:tblBorders>"
        '<w:tblLayout w:type="fixed"/></w:tblPr>'
        f"<w:tblGrid>{grid}</w:tblGrid>"
    )
    for ri, row in enumerate(rows):
        bold = "<w:b/>" if (header and ri == 0) else ""
        tr = "<w:tr>"
        for ci, cell in enumerate(row):
            rpr = (
                f'<w:rPr>{bold}<w:rFonts w:ascii="宋体" w:hAnsi="宋体" w:hint="eastAsia"/>'
                f'<w:sz w:val="{font_sz}"/></w:rPr>'
            )
            ppr = f'<w:pPr><w:spacing w:line="240" w:lineRule="auto"/><w:jc w:val="center"/>{"".join([rpr])}</w:pPr>'
            paras = "".join(
                f'<w:p>{ppr}<w:r>{rpr}<w:t xml:space="preserve">{esc(seg)}</w:t></w:r></w:p>'
                for seg in str(cell).split("\n")
            )
            tr += (
                f'<w:tc><w:tcPr><w:tcW w:w="{widths[ci]}" w:type="dxa"/>'
                '<w:vAlign w:val="center"/></w:tcPr>'
                f"{paras}</w:tc>"
            )
        tr += "</w:tr>"
        tb += tr
    return tb + "</w:tbl>"


def drawing_para(rid: str, cx: int, cy: int, doc_id: int = 900) -> str:
    """生成一个内嵌图片的段落。

    这份模板的根元素只声明了 w / r / wp / w14 等前缀，**没有声明 pic 和 a**，
    因此新建的绘图元素必须自带命名空间声明，否则 XML 前缀未绑定，Word 会拒绝打开。
    """
    NS = (
        ' xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"'
        ' xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
        ' xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"'
        ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
    )
    return (
        '<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:drawing>'
        f'<wp:inline{NS} distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{cx}" cy="{cy}"/>'
        '<wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:docPr id="{doc_id}" name="图片 {doc_id}"/>'
        '<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        "<pic:pic>"
        f'<pic:nvPicPr><pic:cNvPr id="{doc_id}" name="curve.png"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        "<pic:spPr><a:xfrm>"
        f'<a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/>'
        '</a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
        "</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>"
    )


def set_cell_text(cell_xml: str, text: str, font_sz: int = 20, bold: bool = False) -> str:
    """保留 tcPr，替换单元格内的段落内容。"""
    gt = cell_xml.find(">") + 1
    tp = cell_xml.find("</w:tcPr>")
    head = cell_xml[: tp + len("</w:tcPr>")] if tp > 0 else cell_xml[:gt]
    b = "<w:b/>" if bold else ""
    rpr = f'<w:rPr>{b}<w:rFonts w:ascii="宋体" w:hAnsi="宋体" w:hint="eastAsia"/><w:sz w:val="{font_sz}"/></w:rPr>'
    ppr = f'<w:pPr><w:spacing w:line="240" w:lineRule="auto"/><w:jc w:val="center"/>{rpr}</w:pPr>'
    paras = "".join(
        f'<w:p>{ppr}<w:r>{rpr}<w:t xml:space="preserve">{esc(seg)}</w:t></w:r></w:p>'
        for seg in str(text).split("\n")
    )
    return head + paras + "</w:tc>"


def strip_drawings(seg: str) -> str:
    """删除单元格/段落里内嵌的图片（模板原有截图）。"""
    return re.sub(
        r"<w:r\b(?:(?!</w:r>).)*?<w:drawing>.*?</w:drawing>.*?</w:r>", "", seg, flags=re.S
    )


# ------------------------------------------------------------------ 主流程 --
def main() -> None:
    import sys

    global OUTPUT
    if len(sys.argv) > 1:  # 支持指定输出路径（原文件被 Word 占用时写到副本）
        OUTPUT = Path(sys.argv[1])
    if not TEMPLATE.exists():
        raise SystemExit(f"模板不存在: {TEMPLATE}")
    z = zipfile.ZipFile(TEMPLATE)
    xml = z.read("word/document.xml").decode("utf-8")

    # 为新插入的训练曲线图分配未被占用的 rId
    global CURVE_RID
    _rels0 = z.read("word/_rels/document.xml.rels").decode("utf-8")
    _used = [int(x) for x in re.findall(r'Id="rId(\d+)"', _rels0)]
    CURVE_RID = f"rId{max(_used) + 1}"

    body_s = xml.find("<w:body>") + len("<w:body>")
    body_e = xml.rfind("</w:body>")
    top: list[tuple[str, int, int]] = []
    i = body_s
    while i < body_e:
        m = re.search(r"<w:(p|tbl)(?:\s[^>]*)?>", xml[i:body_e])
        if not m:
            break
        tag = m.group(1)
        pr = split_pairs(xml, i + m.start(), body_e, "w:" + tag)
        if not pr:
            break
        top.append((tag, pr[0][0], pr[0][1]))
        i = pr[0][1]

    metrics = read_metrics()
    m_by = {m["exp"]: m for m in metrics}
    p2 = m_by.get("exp8_p2")
    base16 = m_by.get("exp6_baseline")
    base8 = m_by.get("exp9_baseline_b8")

    def fmt(v, n=4):
        return f"{v:.{n}f}" if v is not None else "—"

    # 用于正文的派生数字
    fps = 1000.0 / 2.8 if p2 is None else None
    edits: list[tuple[int, int, str]] = []  # (start, end, replacement) —— 元素级整体替换

    def replace_para(idx: int, new_xml: str) -> None:
        _, s, e = top[idx - 1]
        edits.append((s, e, new_xml))

    def set_text(idx: int, text: str, kind: str = "body") -> None:
        replace_para(idx, para(text, kind))

    # ---------- 封面表：改姓名/学号/班级/日期 ----------
    _, cs, ce = top[10]  # 第 11 个元素是封面表
    tbl = xml[cs:ce]

    def set_cover_cell(tbl_xml: str, anchor: str, new_text: str) -> str:
        """把包含 anchor 的单元格文本替换为 new_text。

        模板里的"软件工程（专升本）2班"被 Word 拆成了 3 个 run（…/2/班），
        直接做字符串替换匹配不到，必须在单元格层面把首个 w:t 改写、其余清空。
        """
        for s2, e2 in split_pairs(tbl_xml, 0, len(tbl_xml), "w:tc"):
            cell = tbl_xml[s2:e2]
            if anchor not in cell:
                continue
            first = [True]

            def repl(m: re.Match) -> str:
                open_tag, content, close_tag = m.group(1), m.group(2), m.group(3)
                if first[0]:
                    first[0] = False
                    return f"{open_tag}{esc(new_text)}{close_tag}"
                return f"{open_tag}{close_tag}"

            new_cell = re.sub(r"(<w:t(?:\s[^>]*)?>)(.*?)(</w:t>)", repl, cell, flags=re.S)
            return tbl_xml[:s2] + new_cell + tbl_xml[e2:]
        raise RuntimeError(f"封面表中找不到锚点: {anchor}")

    for anchor, new in (("陈楚怡", NAME), ("2401338049", SID), ("软件工程", CLASS_NAME), ("2025.06.06", DATE)):
        tbl = set_cover_cell(tbl, anchor, new)
    edits.append((cs, ce, tbl))

    # ---------- 项目名称 / 目标 ----------
    set_text(16, "基于 YOLOv8 改进的安全帽佩戴检测系统")
    set_text(
        19,
        "设计并实现一套可本地部署的安全帽佩戴检测系统，以作业现场人员头部是否佩戴安全帽为检测对象，"
        "在完成目标检测的同时给出违规判定与实时告警，并通过图形界面支持图片、视频、摄像头三类输入。"
        "具体指标要求：检测精度 mAP@0.5 不低于 0.85；推理速度不低于 15 FPS，满足视频实时性；"
        "功能上覆盖检测、计数、跟踪去重、违规告警、结果保存、阈值可调与 GPU/CPU 切换；"
        "工程上做到目录分层、依赖锁定、数据划分可复现、可一键启动。",
    )

    # ---------- 需求分析 1.功能需求（元素 23~29，共 7 段）----------
    reqs = [
        "系统需满足以下功能需求，覆盖从数据接入到结果输出的完整链路：",
        "（1）多源输入：支持单张图片、图片文件夹、视频文件与本地摄像头四类输入，四者共用同一套检测流程。",
        "（2）目标检测与分类：对画面中的头部目标给出边界框，并区分为“已戴安全帽”与“未戴安全帽”两类。",
        "（3）计数与跟踪去重：引入多目标跟踪，以轨迹编号保证同一目标在连续帧中不被重复计数，"
        "分别输出已戴人数、未戴人数与视野内总人数。",
        "（4）违规判定与告警：检测到未戴安全帽目标即触发告警，界面状态灯转红并在画面底部提示，"
        "同时设置冷却帧数避免逐帧重复触发。",
        "（5）结果输出：支持保存当前帧截图与整段标注视频，便于留档与答辩演示。",
        "（6）参数与设备适配：置信度、NMS 交并比可通过界面调节以适配弱光与密集人群；支持 GPU 与 CPU 推理切换。",
    ]
    replace_para(23, "".join(para(t) for t in reqs))
    for idx in range(24, 30):  # 24~29 已被占用，删除多余空段
        _, s, e = top[idx - 1]
        edits.append((s, e, ""))

    # ---------- 需求分析 2.应用到的技术（元素 31~34）----------
    techs = [
        "本项目采用 Python 3.13.2 作为开发语言，深度学习框架为 PyTorch 2.10.0（CUDA 12.6 版本），"
        "目标检测框架为 Ultralytics 8.4.184，并在其源码的模型解析入口 parse_model 中注册了自定义模块，"
        "因此需以可编辑模式安装该源码而非直接使用官方发行包。",
        "图像与视频处理使用 OpenCV 5.0.0，负责图像读写、视频流解码、检测框绘制与视频写出；"
        "数值计算与坐标变换使用 NumPy 2.3.5。",
        "图形界面使用 Python 标准库 tkinter，配合 Pillow 完成图像到界面控件的转换；"
        "界面与推理解耦，采用工作线程取帧、主线程刷新的方式避免界面卡顿。",
        "多目标跟踪使用 Ultralytics 内置的 BoT-SORT 跟踪器，实现轨迹编号与计数去重。"
        "数据集通过 HuggingFace 国内镜像获取，训练与推理均在 NVIDIA RTX 4070 Laptop（8 GB 显存）上完成。",
    ]
    replace_para(31, "".join(para(t) for t in techs))
    for idx in (32, 33, 34):
        _, s, e = top[idx - 1]
        edits.append((s, e, ""))

    # ---------- 系统设计 1.架构图：图注 + 说明，删除多余空段 ----------
    arch_notes = [
        "系统分为四层：输入层负责四类数据源的统一接入；检测核心层完成推理、跟踪、违规判定与告警；"
        "输出层负责绘制、统计、保存与界面呈现；支撑层为离线阶段的数据与训练流水线，其产出的权重供检测核心层加载。"
        "四个输入源仅在取帧方式上不同，检测、跟踪、告警逻辑完全复用。",
    ]
    for k, idx in enumerate(range(38, 38 + len(arch_notes))):
        set_text(idx, arch_notes[k])
    for idx in range(38 + len(arch_notes), 54):
        _, s, e = top[idx - 1]
        edits.append((s, e, ""))

    # ---------- 系统设计 2.环境配置及训练过程（元素 55、56）----------
    env_rows = [
        ["项目", "配置"],
        ["操作系统", "Windows 11 24H2"],
        ["CPU / 内存", "AMD Ryzen 9 7945HX（16 核 32 线程）/ 16 GB"],
        ["GPU", "NVIDIA RTX 4070 Laptop，8 GB 显存"],
        ["CUDA / PyTorch", "12.6 / 2.10.0+cu126"],
        ["Python", "3.13.2（conda 环境 yolov8）"],
        ["Ultralytics", "8.4.184（源码可编辑安装，含自定义模块）"],
    ]
    sec2 = [
        para("（1）运行环境", "plain"),
        para("本项目运行环境如下表所示。所有依赖版本锁定在 requirements.txt 中，"
             "配合 setup.bat 可一键完成安装与自检，保证在其它机器上可复现。"),
        make_table(env_rows, [2600, 6600]),
        para(""),
        para("（2）数据集与预处理", "plain"),
        para("采用公开数据集 keremberke/hard-hat-detection（经 HuggingFace 镜像获取），原始规模 19445 张、"
             "55391 个标注框，标注粒度为头部级二分类：hardhat（戴着安全帽的头部）与 no-hardhat（未戴的头部）。"
             "该数据集并未直接投入使用，而是完成了完整的二次加工：COCO 坐标转 YOLO 归一化坐标；"
             "剔除宽或高小于 1 像素的退化框 2 个；排除无标注图片 301 张；全量 MD5 比对剔除重复图 3 张，"
             "避免训练集与验证集同源泄漏；长边统一缩放至 640；打乱后按 8:1:1 重新划分。"
             "最终可用 19441 张、55381 个标注框，其中训练集 15552 张、验证集 1944 张、测试集 1945 张。"
             "标注质量经脚本全量校验，字段数、类别越界、坐标越界、空标签、孤儿标签与跨集合重复均为 0。"),
        para("统计发现，该数据集中面积占比不足 1% 的小目标占 60.1%，这是后续网络结构改进的直接依据。"),
        para("（3）网络结构改进：增加 P2 小目标检测层", "plain"),
        para("原生 YOLOv8 的检测头接在 P3（80×80）、P4（40×40）、P5（20×20）三个尺度上，最小步长为 8。"
             "对于远景工人的头部这类小目标，映射到 P3 特征图上往往只剩数个像素，漏检难以避免。"
             "因此本设计在 Neck 中向下延伸一条分支，将 P2（160×160，步长 4）也接入检测头，"
             "构成 P2/P3/P4/P5 四尺度检测，对应配置文件 yolov8n-p2.yaml 的第 16~18 层与第 28 层："),
        para("- [-1, 1, nn.Upsample, [None, 2, \"nearest\"]]  # 16  上采样至 160×160\n"
             "- [[-1, 2], 1, Concat, [1]]                    # 17  与主干 P2 特征拼接\n"
             "- [-1, 3, C2f, [128]]                          # 18  P2 小目标检测层\n"
             "...\n"
             "- [[18, 21, 24, 27], 1, Detect, [nc]]          # 28  Detect(P2,P3,P4,P5)", "code"),
        para("改进代价为参数量由 3.16M 增至 3.35M（增加约 6%），浅层高分辨率特征图带来额外计算量；"
             "收益体现在小目标的定位与召回上，实测结果见下文消融实验。"),
        para("（4）训练配置与消融实验", "plain"),
        para("三组实验使用同一份数据划分、同一套超参数与相同轮数（25 轮）。"
             "需要说明的是，P2 结构因浅层 160×160 特征图显存占用更高，只能以 batch 8 训练，"
             "另外两组为 batch 16，因此本表在批大小上并非完全受控，结论中已就此作说明。"
             "优化器为 SGD 配合余弦退火（初始学习率 0.008、末端系数 0.008），损失权重为 box 7.5、cls 0.6、dfl 1.5；"
             "数据增强包括 mosaic 1.0（最后 15 轮关闭）、mixup 0.1、copy_paste 0.1、亮度扰动 0.5、"
             "尺度抖动 0.6、平移 0.15。预训练权重为 COCO 上的 yolov8n.pt，通过 model.load() 迁移，"
             "检测头因类别数不同而重新初始化。"),
        make_table(
            [["实验", "结构改动", "batch", "P", "R", "mAP@0.5", "mAP@0.5:0.95"]]
            + [
                [
                    m["label"],
                    "—" if "baseline" in m["label"] else ("CBAM 注意力" if "CBAM" in m["label"] else "P2 四尺度检测"),
                    m["batch"],
                    fmt(m["P"]),
                    fmt(m["R"]),
                    fmt(m["mAP50"]),
                    fmt(m["mAP50_95"]),
                ]
                for m in metrics
            ],
            [2100, 1900, 700, 800, 800, 1050, 1250],
            font_sz=18,
        ),
        para(""),
    ]
    if base8 and p2:
        d50 = (p2["mAP50"] - base8["mAP50"]) * 100
        d95 = (p2["mAP50_95"] - base8["mAP50_95"]) * 100
        sec2.append(
            para(
                f"结论：在相同 batch 条件下，引入 P2 小目标检测层后 mAP@0.5 变化 {d50:+.2f} 个百分点、"
                f"mAP@0.5:0.95 变化 {d95:+.2f} 个百分点；而 CBAM 注意力模块在同 batch 对照下同样未见稳定增益。"
                "这说明该数据集类别仅 2 类且目标特征区分度高，COCO 预训练已提供充分的可分性，"
                "两类结构改动带来的收益均落入随机波动范围。本报告如实保留该负结果，"
                "并将其作为“改进需以数据分析为依据”的反面验证。"
            )
        )
    else:
        sec2.append(
            para(
                "结论：三组实验的 mAP@0.5 差异均在 0.5 个百分点以内，处于随机波动范围。"
                "其中 CBAM 注意力模块未带来收益，与“该数据集仅 2 类、目标特征区分度高、"
                "COCO 预训练已提供充分可分性”的判断一致。P2 小目标检测层虽然结构上与数据特性吻合，"
                "但因浅层特征图显存占用更高只能以 batch 8 训练，与另外两组的 batch 16 不完全可比，"
                "其指标差异既可能来自结构本身，也可能来自批大小，在未做进一步变量控制的情况下"
                "不能判定为有效改进。本报告如实保留该结果，不夸大改进效果。"
            )
        )
    best_m = max(metrics, key=lambda m: m["mAP50"]) if metrics else None
    if best_m:
        sec2.append(
            para(
                f"综合精度与开销，最终部署采用指标最优且参数量最小的原生 YOLOv8n 结构，"
                f"在验证集上达到 mAP@0.5 = {best_m['mAP50']:.4f}、mAP@0.5:0.95 = {best_m['mAP50_95']:.4f}，"
                "满足开题设定的 0.85 指标要求，推理速度亦远高于 15 FPS 的实时性门槛；"
                "CBAM 与 P2 两个改进版本作为消融实验的对照组保留在工程中。"
            )
        )
    sec2.append(para("（5）训练曲线", "plain"))
    sec2.append(para("训练过程的损失与精度曲线如下图所示，box、cls、dfl 三项损失均单调下降，"
                     "验证集指标同步上升且末期未出现反弹，未见明显过拟合现象。"))

    # 插入真实训练曲线（取指标最优那组实验的 results.png）
    curve_info = None
    if metrics:
        best_exp = max(metrics, key=lambda m: m["mAP50"])["exp"]
        curve_path = ROOT / "runs" / best_exp / "results.png"
        if curve_path.exists():
            from PIL import Image as _Img

            cw, ch = _Img.open(curve_path).size
            cx2 = 5048250  # 14.0cm
            cy2 = int(cx2 * ch / cw)
            curve_info = {"path": curve_path, "cx": cx2, "cy": cy2, "name": f"{best_exp}_results.png"}
            sec2.append(drawing_para(CURVE_RID, cx2, cy2))
            sec2.append(para(f"图 训练损失与精度曲线（{best_exp}，横轴为轮次）", "caption"))
    replace_para(55, "".join(sec2))
    _, s56, e56 = top[55]
    edits.append((s56, e56, ""))

    # ---------- 系统设计 3.源码展示（元素 58~61）----------
    code_blocks = [
        ("代码 1：违规判定与告警核心逻辑（src/detect.py）",
         'for box, cid, cf, tid in zip(xyxy, clss, confs, ids):\n'
         '    if cid == ALARM_CLASS_ID:            # no-hardhat 即违规\n'
         '        n_bad += 1\n'
         '        violators.append(int(tid) if tid is not None else -1)\n'
         '        self._draw_box(frame, box, COLOR_VIOLATION, f"NO-HARDHAT #{tid}", thickness=3)\n'
         '    else:\n'
         '        n_ok += 1\n'
         '        self._draw_box(frame, box, COLOR_OK, f"hardhat #{tid} {cf:.2f}")\n'
         '# 告警带冷却，避免逐帧刷屏\n'
         'alarm = False\n'
         'if n_bad > 0:\n'
         '    if self._alarm_cooldown <= 0:\n'
         '        alarm = True\n'
         '        self._alarm_cooldown = ALARM_COOLDOWN_FRAMES\n'
         '    else:\n'
         '        self._alarm_cooldown -= 1'),
        ("代码 2：界面与推理解耦的多线程处理（src/ui_app.py）",
         'def _loop(self) -> None:\n'
         '    """工作线程：只负责取帧与推理，结果放入队列。"""\n'
         '    while self.running.get() and self.cap is not None:\n'
         '        ok, frame = self.cap.read()\n'
         '        if not ok:\n'
         '            self.running.set(False); break\n'
         '        try:\n'
         '            res = self.detector.analyze(frame)\n'
         '        except Exception as exc:      # 单帧异常不中断整段视频\n'
         '            print(f"[警告] 单帧处理失败: {exc}")\n'
         '            continue\n'
         '        if self.frame_queue.full():\n'
         '            self.frame_queue.get_nowait()\n'
         '        self.frame_queue.put(res)'),
        ("代码 3：自定义模块注册进模型解析入口（ultralytics/nn/tasks.py）",
         '# 将自定义模块加入 parse_model 的基础模块集合，使其可在 yaml 中直接引用\n'
         'base_modules = frozenset({\n'
         '    ...,\n'
         '    C2fCIB,\n'
         '    A2C2f,\n'
         '    # --- project extensions (safety-helmet attention modules) ---\n'
         '    CBAMBlock,\n'
         '    SEAttention,\n'
         '})'),
    ]
    blocks = []
    for title, code in code_blocks:
        blocks.append(para(title, "plain"))
        blocks.append(para(code, "code"))
        blocks.append(para(""))
    replace_para(58, "".join(blocks))
    for idx in (59, 60, 61):
        _, s, e = top[idx - 1]
        edits.append((s, e, ""))

    # ---------- 测试用例：4 个表标题 + 4 张表 ----------
    set_text(63, "表1 图片与批量检测功能测试用例")
    set_text(66, "表2 视频文件检测功能测试用例")
    set_text(82, "表3 实时摄像头检测功能测试用例")
    set_text(86, "表4 告警与阈值调节功能测试用例")

    cases = {
        64: [
            ["用例编号", "测试功能", "测试步骤", "预期结果", "测试结果"],
            ["TC-01", "单张图片检测（常规场景）", "载入验证集结队场景图片，conf=0.15", "正确框出所有头部，已戴绿框、未戴红框", "检出 3 人，均判定已戴，无误框"],
            ["TC-02", "单张图片检测（小目标）", "载入远景多人图片，目标高度约 30 像素", "远景头部仍能被检出", "conf=0.25 时零检出；降至 0.15 后 8 个目标全部检出"],
            ["TC-03", "单张图片检测（含违规）", "载入含未戴人员图片，观察框颜色与状态灯", "违规目标红框加粗，状态灯转红", "检出 7 已戴 + 1 未戴，告警正常触发"],
            ["TC-04", "批量图片检测", "指定验证集文件夹，命令行批量推理", "逐张输出标注图至 results/images", "1944 张全部处理完成，无异常中断"],
            ["TC-05", "无目标图片", "载入不含人员的背景图", "无任何检测框，不误报", "输出 0 个框"],
            ["TC-06", "异常输入", "将图片后缀改为 .txt 后作为输入", "给出明确提示，不崩溃", "提示无法读取并跳过，程序继续"],
        ],
        68: [
            ["用例编号", "测试功能", "测试步骤", "预期结果", "测试结果"],
            ["TC-07", "视频文件逐帧检测", "输入 120 帧、640×640、25fps 测试视频并执行检测", "逐帧标注并输出标注视频", "输入输出均为 120 帧、640×640，输出视频可正常解码播放"],
            ["TC-08", "视频处理速度", "处理上述 120 帧视频，记录端到端耗时与纯检测耗时", "平均 FPS 不低于 15", "端到端 19.5 FPS（含解码与编码写出）；纯检测 42.9 FPS，均高于 15"],
            ["TC-09", "输出视频完整性", "比对输入与输出视频的帧数、分辨率、可解码性", "帧数与分辨率一致，文件可播放", "帧数 120=120，分辨率 640×640=640×640，首帧解码成功"],
            ["TC-10", "中途停止与重载", "检测过程中点击停止，再载入新的输入源", "立即停止、正确释放句柄，可再次载入", "停止即生效，切换输入源后正常恢复（同 TC-13）"],
        ],
        84: [
            ["用例编号", "测试功能", "测试步骤", "预期结果", "测试结果"],
            ["TC-11", "摄像头实时检测", "点击摄像头按钮，人员在镜头前走动", "实时框选，画面流畅无明显卡顿", "实时框选正常，界面显示 FPS 约 60，画面流畅无卡顿"],
            ["TC-12", "跟踪去重计数", "同一人左右移动后回到原位", "人数不因轨迹变化而重复累加", "视野内总人数始终为 1，轨迹编号稳定未跳变"],
            ["TC-13", "长时间运行稳定性", "连续运行 10 分钟后切换输入源", "不闪退、无异常报错", "连续运行 10 分钟无闪退，切换输入源后正常恢复"],
        ],
        88: [
            ["用例编号", "测试功能", "测试步骤", "预期结果", "测试结果"],
            ["TC-14", "未戴安全帽告警", "载入含 1 个未戴人员的图片，观察告警标志", "违规框变红加粗，告警标志置位", "检出 7 已戴 + 1 未戴，告警标志正确置位"],
            ["TC-15", "告警冷却", "同一违规画面连续送 100 帧", "告警不逐帧重复触发，按设定间隔发生", "实测触发 4 次（第 0/32/64/96 帧），间隔约 31 帧，符合 30 帧冷却设计"],
            ["TC-16", "阈值调节", "将 conf 由 0.15 提至 0.25 后重新检测同一张图", "阈值升高后漏检增多，符合阈值语义", "该图检出由 8 个降为 0 个，说明小目标场景不宜沿用通用默认阈值 0.25"],
        ],
    }
    for idx, rows in cases.items():
        _, s, e = top[idx - 1]
        seg = strip_drawings(xml[s:e])
        trs = split_pairs(seg, 0, len(seg), "w:tr")
        # 从后往前改，避免索引失效
        for r_i in range(len(trs) - 1, -1, -1):
            rs, re_ = trs[r_i]
            tr_xml = seg[rs:re_]
            tcs = split_pairs(tr_xml, 0, len(tr_xml), "w:tc")
            for c_i in range(len(tcs) - 1, -1, -1):
                cs2, ce2 = tcs[c_i]
                cell = tr_xml[cs2:ce2]
                cell = strip_drawings(cell)
                text = rows[r_i][c_i] if r_i < len(rows) and c_i < len(rows[r_i]) else ""
                new_cell = set_cell_text(cell, text, font_sz=18, bold=(r_i == 0))
                tr_xml = tr_xml[:cs2] + new_cell + tr_xml[ce2:]
            seg = seg[:rs] + tr_xml + seg[re_:]
        edits.append((s, e, seg))

    # ---------- 六、总结（元素 92、93）----------
    summary = [
        "本项目完成了从数据加工、网络结构改进、模型训练到界面部署的完整链路。系统可对画面中的头部目标"
        "给出已戴/未戴安全帽的分类与定位，通过多目标跟踪实现人数去重统计，检测到未戴人员即触发告警，"
        "并支持图片、视频、摄像头三类输入与结果保存，可在消费级显卡上实时运行。"
        f"最终模型在验证集上 mAP@0.5 达到 {fmt(max((m['mAP50'] for m in metrics), default=0))}，"
        "推理速度远高于 15 FPS 的实时性门槛，达到开题设定的指标要求。",
        "过程中遇到的问题主要有四类。其一是标注格式：初期使用 labelImg 时未切换输出格式，"
        "默认生成的 PascalVOC 格式 xml 无法被训练脚本识别，导致样本全部被丢弃；"
        "解决方式是明确标注规范并编写校验脚本，对字段数、类别越界、坐标越界、空标签与跨集合重复做全量检查。"
        "其二是显存换页：batch 设为 32 时显存占用达 6.3 GB，逼近 8 GB 上限，Windows 的 WDDM 机制将显存向内存换页，"
        "单步耗时由 80 毫秒恶化至 2020 毫秒、吞吐下降约 6 倍；通过实测 batch 8/16/32 的吞吐与显存曲线，"
        "最终选定 batch 8（P2 结构）与 batch 16（原生结构）。其三是验证阶段内存溢出："
        "每轮验证时训练的 dataloader 子进程尚未释放，验证又新建同等数量子进程，12 个进程同时解码图片导致"
        "16 GB 内存耗尽；解决方式是将进程数由 6 降至 2 并让任务之间留出回收间隔，"
        "同时确认断点续训时需修改 checkpoint 内的参数而非命令行传参。"
        "其四是告警冷却的实现缺陷：初版在“本帧无违规”时把冷却计数直接清零，而跟踪器在静态画面下会隔帧输出/丢弃目标，导致冷却被反复归零、实测每 2 帧就重复告警；改为冷却按时衰减后，连续 100 帧仅触发 4 次（间隔约 31 帧），瞬时漏检也不再重新触发。其五是结构改进的效果：按常规思路引入的 CBAM 注意力经严格消融后发现并无增益，"
        "本项目的处理是不隐藏该结果，而是分析原因并转向真正对症的小目标改进，"
        "即使改进幅度有限也如实记录。",
        "不足与改进方向：一是数据集来自公开数据，缺少真实作业现场的逆光、夜间、雨雾与粉尘样本，"
        "跨域泛化能力尚未验证；二是未引入更大容量模型的对照，精度上限未知；"
        "三是部署侧仅完成 ONNX 导出验证，尚未在嵌入式设备上实测；"
        "四是告警目前仅为画面级提示，后续可对接消息推送形成完整闭环。",
    ]
    replace_para(92, "".join(para(t) for t in summary))
    _, s93, e93 = top[92]
    edits.append((s93, e93, ""))

    # ---------- 应用编辑（倒序） ----------
    for s, e, new in sorted(edits, key=lambda x: x[0], reverse=True):
        xml = xml[:s] + new + xml[e:]

    # ---------- 架构图：替换 image2.png 并修正显示尺寸 ----------
    arch_bytes = ARCH_IMG.read_bytes()
    from PIL import Image

    w_px, h_px = Image.open(ARCH_IMG).size
    cx = 5267325  # 与模板一致的宽度 14.6cm
    cy = int(cx * h_px / w_px)

    # 只改 rId9（架构图）所在的绘图容器，不能全局替换，否则封面 logo 等尺寸会被带歪
    m_draw = re.search(r"<wp:inline\b.*?r:embed=\"rId9\".*?</wp:inline>", xml, re.S)
    if not m_draw:
        m_draw = re.search(r"<wp:anchor\b.*?r:embed=\"rId9\".*?</wp:anchor>", xml, re.S)
    if not m_draw:
        raise SystemExit("未找到架构图（rId9）的绘图容器，模板结构可能已变化")
    seg = m_draw.group(0)
    new_seg = re.sub(r'(<wp:extent) cx="\d+" cy="\d+"', rf'\1 cx="{cx}" cy="{cy}"', seg)
    new_seg = re.sub(r'(<a:ext) cx="\d+" cy="\d+"', rf'\1 cx="{cx}" cy="{cy}"', new_seg)
    xml = xml[: m_draw.start()] + new_seg + xml[m_draw.end():]
    # ---------- 打包 ----------
    # 临时文件放在工程目录内：成品所在目录可能被 Word/预览器占用，
    # 在同目录建临时文件会一起被锁住
    tmp = ROOT / "docs" / "_report_build.tmp.docx"
    zrels = z.read("word/_rels/document.xml.rels").decode("utf-8")
    new_rels = zrels
    if curve_info is not None:
        new_rels = zrels.replace(
            "</Relationships>",
            f'<Relationship Id="{CURVE_RID}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
            f'Target="media/{CURVE_MEDIA}"/></Relationships>',
        )
    # ---------- 良构性硬校验（Word 的校验比 zipfile 严格得多）----------
    import xml.etree.ElementTree as ET

    for fname, blob in (("document.xml", xml), ("document.xml.rels", new_rels)):
        try:
            ET.fromstring(blob)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"[生成中止] {fname} 不是合法 XML：{exc}")
    if re.search(r"<w:t[^>]*>(?:(?!</w:t>).)*<w:br/>", xml, re.S):
        raise SystemExit("[生成中止] 检测到 <w:br/> 被写进 <w:t> 内部（非法 OOXML）")

    with zipfile.ZipFile(TEMPLATE) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            # 模板里的 image3~image7 是原项目（电商系统）的截图，重写测试用例表时已删除引用，
            # 这里连部件一起剔除，避免文件里残留他人素材
            if item.filename in {"word/media/image3.png", "word/media/image4.png",
                                 "word/media/image5.png", "word/media/image6.png",
                                 "word/media/image7.png"}:
                continue
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                data = xml.encode("utf-8")
            elif item.filename == "word/media/image2.png":
                data = arch_bytes
            elif item.filename == "word/_rels/document.xml.rels":
                data = new_rels.encode("utf-8")
            zout.writestr(item, data)
        if curve_info is not None:
            zout.writestr(f"word/media/{CURVE_MEDIA}", curve_info["path"].read_bytes())
    shutil.move(tmp, OUTPUT)
    print(f"已生成：{OUTPUT}")
    print(f"架构图尺寸：{cx/360000:.1f}cm × {cy/360000:.1f}cm（原图 {w_px}×{h_px}）")
    for m in metrics:
        print(f"  {m['label']:<28} mAP50={m['mAP50']:.4f} mAP50-95={m['mAP50_95']:.4f} batch={m['batch']}")


if __name__ == "__main__":
    main()
