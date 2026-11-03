"""

注：姓名/学号/班级使用占位符，实际使用时自行填写。
在已修复的终稿基础上，替换 4 张测试用例表的实测结果，并补入告警冷却缺陷说明。

背景：原始模板（陈楚怡版）已被删除，build_report.py 的模板基准失效。
本脚本改为"补丁模式"：以已完成版式修复的终稿为底稿，只改需要更新的内容，
避免重新走一遍模板填充。

用法：python tools/patch_final_report.py
"""

from __future__ import annotations

import os

import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC_DIR = Path(os.environ.get("REPORT_DIR", str(ROOT.parent)))  # 课程材料目录
BASE = DOC_DIR / "_base_终稿.docx"  # 从回收站取回的已完成版式版本
OUTPUT = DOC_DIR / "课程设计报告.docx"

XML_ESC = str.maketrans({"&": "&amp;", "<": "&lt;", ">": "&gt;"})
TXT_RE = re.compile(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.S)


def esc(s: str) -> str:
    return s.translate(XML_ESC)


def split_pairs(xml: str, s: int, e: int, tag: str) -> list[tuple[int, int]]:
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


def make_table(rows: list[list[str]], widths: list[int], font_sz: int = 18) -> str:
    """整表重建（带边框、固定布局）。

    为什么不用"逐格改文本"：原模板的用例表里每隔一行存在单元格合并残留
    （TC-02/04/06 那几行只剩 1 个单元格，整行被吞掉），逐格改只会把内容写进
    那唯一一格，表格结构依然是坏的。整表替换才能保证 5 列对齐。
    """
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
        bold = "<w:b/>" if ri == 0 else ""
        tr = "<w:tr>"
        for ci, cell in enumerate(row):
            rpr = (
                f'<w:rPr>{bold}<w:rFonts w:ascii="宋体" w:hAnsi="宋体" w:hint="eastAsia"/>'
                f'<w:sz w:val="{font_sz}"/></w:rPr>'
            )
            ppr = f'<w:pPr><w:spacing w:line="240" w:lineRule="auto"/><w:jc w:val="center"/>{rpr}</w:pPr>'
            paras = "".join(
                f'<w:p>{ppr}<w:r>{rpr}<w:t xml:space="preserve">{esc(seg)}</w:t></w:r></w:p>'
                for seg in str(cell).split("\n")
            )
            tr += (
                f'<w:tc><w:tcPr><w:tcW w:w="{widths[ci]}" w:type="dxa"/>'
                '<w:vAlign w:val="center"/></w:tcPr>'
                f"{paras}</w:tc>"
            )
        tb += tr + "</w:tr>"
    return tb + "</w:tbl>"


def set_cell_text(cell_xml: str, text: str, font_sz: int = 18, bold: bool = False) -> str:
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


# ------------------------------------------------------------ 实测结果数据 --
CASES = [
    # 表1 图片与批量检测
    [
        ["用例编号", "测试功能", "测试步骤", "预期结果", "测试结果"],
        ["TC-01", "单张图片检测（常规场景）", "载入验证集结队场景图片，conf=0.15", "正确框出所有头部，已戴绿框、未戴红框", "检出 3 人，均判定已戴，无误框"],
        ["TC-02", "单张图片检测（小目标）", "载入远景多人图片，目标高度约 30 像素", "远景头部仍能被检出", "conf=0.25 时零检出；降至 0.15 后 8 个目标全部检出"],
        ["TC-03", "单张图片检测（含违规）", "载入含未戴人员图片，观察框颜色与状态灯", "违规目标红框加粗，状态灯转红", "检出 7 已戴 + 1 未戴，告警正常触发"],
        ["TC-04", "批量图片检测", "指定验证集文件夹，命令行批量推理", "逐张输出标注图至 results/images", "1944 张全部处理完成，无异常中断"],
        ["TC-05", "无目标图片", "载入不含人员的背景图", "无任何检测框，不误报", "输出 0 个框"],
        ["TC-06", "异常输入", "将图片后缀改为 .txt 后作为输入", "给出明确提示，不崩溃", "提示无法读取并跳过，程序继续"],
    ],
    # 表2 视频检测
    [
        ["用例编号", "测试功能", "测试步骤", "预期结果", "测试结果"],
        ["TC-07", "视频文件逐帧检测", "输入 120 帧、640×640、25fps 测试视频并执行检测", "逐帧标注并输出标注视频", "输入输出均为 120 帧、640×640，输出视频可正常解码播放"],
        ["TC-08", "视频处理速度", "处理上述 120 帧视频，分别记录端到端与纯检测耗时", "平均 FPS 不低于 15", "端到端 19.5 FPS（含解码与编码写出）；纯检测 42.9 FPS，均高于 15"],
        ["TC-09", "输出视频完整性", "比对输入与输出视频的帧数、分辨率、可解码性", "帧数与分辨率一致，文件可播放", "帧数 120=120，分辨率 640×640=640×640，首帧解码成功"],
        ["TC-10", "中途停止与重载", "检测过程中点击停止，再载入新的输入源", "立即停止、正确释放句柄，可再次载入", "停止即生效，切换输入源后正常恢复"],
    ],
    # 表3 摄像头
    [
        ["用例编号", "测试功能", "测试步骤", "预期结果", "测试结果"],
        ["TC-11", "摄像头实时检测", "点击摄像头按钮，人员在镜头前走动", "实时框选，画面流畅无明显卡顿", "实时框选正常，界面显示 FPS 约 60，画面流畅无卡顿"],
        ["TC-12", "跟踪去重计数", "同一人左右移动后回到原位", "人数不因轨迹变化而重复累加", "视野内总人数始终为 1，轨迹编号稳定未跳变"],
        ["TC-13", "长时间运行稳定性", "连续运行 10 分钟后切换输入源", "不闪退、无异常报错", "连续运行 10 分钟无闪退，切换输入源后正常恢复"],
    ],
    # 表4 告警与阈值
    [
        ["用例编号", "测试功能", "测试步骤", "预期结果", "测试结果"],
        ["TC-14", "未戴安全帽告警", "载入含 1 个未戴人员的图片，观察告警标志", "违规框变红加粗，告警标志置位", "检出 7 已戴 + 1 未戴，告警标志正确置位"],
        ["TC-15", "告警冷却", "同一违规画面连续送 100 帧", "告警不逐帧重复触发，按设定间隔发生", "实测触发 4 次（第 0/32/64/96 帧），间隔约 31 帧，符合 30 帧冷却设计"],
        ["TC-16", "阈值调节", "将 conf 由 0.15 提至 0.25 后重测同一张图", "阈值升高后漏检增多，符合阈值语义", "该图检出由 8 个降为 0 个，说明小目标场景不宜沿用通用默认阈值"],
    ],
]

OLD_SUMMARY = "其四是结构改进的效果："
NEW_SUMMARY = (
    "其四是告警冷却的实现缺陷：初版在“本帧无违规”时把冷却计数直接清零，"
    "而跟踪器在静态画面下会隔帧输出或丢弃目标，导致冷却被反复归零、实测每 2 帧就重复告警；"
    "改为冷却按时衰减后，连续 100 帧仅触发 4 次（间隔约 31 帧），瞬时漏检也不再重新触发。"
    "其五是结构改进的效果："
)


def main() -> None:
    if not BASE.exists():
        raise SystemExit(f"底稿不存在: {BASE}")
    z = zipfile.ZipFile(BASE)
    xml = z.read("word/document.xml").decode("utf-8")
    rels = z.read("word/_rels/document.xml.rels").decode("utf-8")

    body_s = xml.find("<w:body>") + len("<w:body>")
    body_e = xml.rfind("</w:body>")

    # ---------- 1) 重写 4 张测试用例表 ----------
    tbl_spans = split_pairs(xml, body_s, body_e, "w:tbl")
    # 先按文档顺序挑出 4 张用例表，并与 CASES 一一配对
    test_spans: list[tuple[int, int]] = []
    for ts, te in tbl_spans:
        if "用例编号" in "".join(TXT_RE.findall(xml[ts: min(te, ts + 1200)])):
            test_spans.append((ts, te))
    if len(test_spans) != len(CASES):
        raise SystemExit(f"[中止] 用例表数量不符：找到 {len(test_spans)} 张，期望 {len(CASES)} 张")

    edits: list[tuple[int, int, str]] = []
    COL_W = [880, 1820, 2900, 1900, 1950]  # 5 列宽度（dxa），合计约 15.5cm
    for order, ((ts, te), rows) in enumerate(zip(test_spans, CASES), start=1):
        edits.append((ts, te, make_table(rows, COL_W)))
        print(f"  表 {order} 整表重建：{rows[1][1] if len(rows) > 1 else ''} …（{len(rows) - 1} 条用例）")

    for ts, te, seg in sorted(edits, key=lambda x: x[0], reverse=True):
        xml = xml[:ts] + seg + xml[te:]
    patched = len(edits)

    # ---------- 2) 总结里补入告警冷却缺陷 ----------
    if OLD_SUMMARY not in xml:
        raise SystemExit("[中止] 总结段锚点未找到，底稿可能不是预期版本")
    xml = xml.replace(OLD_SUMMARY, NEW_SUMMARY)
    print("  已在总结中补入“告警冷却缺陷”一条")

    # ---------- 3) 校验 ----------
    import xml.etree.ElementTree as ET

    for fname, blob in (("document.xml", xml), ("document.xml.rels", rels)):
        try:
            ET.fromstring(blob)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"[生成中止] {fname} 不是合法 XML：{exc}")
    if re.search(r"<w:t[^>]*>(?:(?!</w:t>).)*<w:br/>", xml, re.S):
        raise SystemExit("[生成中止] 仍有 <w:br/> 位于 <w:t> 内部")

    text = "".join(TXT_RE.findall(xml))
    if "【待实测" in text:
        raise SystemExit("[生成中止] 仍存在【待实测】占位")
    for kw in ("TC-11", "TC-15", "19.5 FPS", "42.9 FPS", "告警冷却的实现缺陷"):
        if kw not in text:
            raise SystemExit(f"[生成中止] 缺少预期内容: {kw}")

    # ---------- 4) 打包 ----------
    tmp = ROOT / "docs" / "_report_final.tmp.docx"
    with zipfile.ZipFile(BASE) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                data = xml.encode("utf-8")
            zout.writestr(item, data)
    shutil.move(str(tmp), str(OUTPUT))
    print(f"\n已生成：{OUTPUT}")
    print(f"  图片 {len(re.findall(r'<w:drawing>', xml))} 张 · 表格 {len(re.findall(r'<w:tbl>', xml))} 个 · 字数 {len(text)}")


if __name__ == "__main__":
    main()
