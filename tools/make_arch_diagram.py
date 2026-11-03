"""生成《系统功能模块架构图》—— 供课程设计报告"系统设计"一节使用。

用 PIL 直接绘制（不依赖 graphviz/matplotlib 的中文字体配置），输出 1800x1100 PNG。
配色沿用国内习惯：主色蓝、告警红、合规绿。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "docs" / "系统功能模块架构图.png"
FONT = "C:/Windows/Fonts/msyh.ttc"
FONT_B = "C:/Windows/Fonts/msyhbd.ttc"

W, H = 1800, 1120
BG = (255, 255, 255)
C_TITLE = (32, 48, 80)
C_INPUT = (232, 241, 253)
C_INPUT_BD = (66, 133, 244)
C_CORE = (235, 248, 240)
C_CORE_BD = (52, 168, 83)
C_OUT = (253, 244, 232)
C_OUT_BD = (245, 150, 40)
C_SUP = (243, 240, 250)
C_SUP_BD = (130, 110, 190)
C_TEXT = (30, 30, 30)
C_GRAY = (120, 120, 120)


def f(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_B if bold else FONT, size)


def box(d: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, title: str, lines: list[str],
        fill: tuple, border: tuple, title_size: int = 30, body_size: int = 22) -> None:
    d.rounded_rectangle([x, y, x + w, y + h], radius=14, fill=fill, outline=border, width=3)
    ft = f(title_size, bold=True)
    tw = d.textlength(title, font=ft)
    d.text((x + (w - tw) / 2, y + 12), title, font=ft, fill=C_TEXT)
    fb = f(body_size)
    for i, ln in enumerate(lines):
        lw = d.textlength(ln, font=fb)
        d.text((x + (w - lw) / 2, y + 14 + title_size + 10 + i * (body_size + 10)), ln, font=fb, fill=C_GRAY)


def arrow(d: ImageDraw.ImageDraw, x1: int, y1: int, x2: int, y2: int, color=(110, 110, 110), width=3) -> None:
    d.line([x1, y1, x2, y2], fill=color, width=width)
    # 箭头
    import math

    ang = math.atan2(y2 - y1, x2 - x1)
    size = 12
    for s in (-1, 1):
        d.line(
            [x2, y2, x2 - size * math.cos(ang - s * 0.5), y2 - size * math.sin(ang - s * 0.5)],
            fill=color,
            width=width,
        )


def main() -> None:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    title = "系统功能模块架构图"
    ft = f(44, bold=True)
    d.text(((W - d.textlength(title, font=ft)) / 2, 24), title, font=ft, fill=C_TITLE)
    sub = "基于 YOLOv8 + CBAM 的安全帽佩戴检测系统"
    fs = f(24)
    d.text(((W - d.textlength(sub, font=fs)) / 2, 82), sub, font=fs, fill=C_GRAY)
    d.line([120, 126, W - 120, 126], fill=(210, 215, 225), width=2)

    # ---------------- 输入层 ----------------
    d.text((120, 150), "① 输入层", font=f(28, bold=True), fill=C_INPUT_BD)
    inputs = ["单张图片", "图片文件夹", "视频文件", "实时摄像头"]
    bw, gap = 330, 44
    for i, name in enumerate(inputs):
        box(d, 120 + i * (bw + gap), 195, bw, 96, name, [], C_INPUT, C_INPUT_BD, 28, 20)

    # ---------------- 检测核心层 ----------------
    core = [
        ("YOLOv8n + CBAM", ["backbone/neck 4 处注意力", "输入 640×640"]),
        ("BoT-SORT 跟踪", ["卡尔曼预测 + 匈牙利匹配", "track id 去重计数"]),
        ("违规判定", ["class=1 (no-hardhat)", "即判为违规目标"]),
        ("告警 / 计数", ["已戴/未戴/总人数", "30 帧冷却防刷屏"]),
    ]
    cw, cg = 380, 34
    for i, (t, ls) in enumerate(core):
        box(d, 120 + i * (cw + cg), 375, cw, 150, t, ls, C_CORE, C_CORE_BD, 28, 21)

    # 输入层 → 核心层（先画箭头，再用白底标签盖住，避免文字被箭头穿过）
    for i in range(4):
        arrow(d, 120 + i * (bw + gap) + bw // 2, 291, 120 + i * (cw + cg) + cw // 2, 372)
    d.rectangle([112, 318, 560, 352], fill=BG)
    d.text((120, 320), "② 检测核心层（src/detect.py）", font=f(28, bold=True), fill=C_CORE_BD)

    # 核心层内部横向流程
    for i in range(3):
        arrow(d, 120 + i * (cw + cg) + cw, 450, 120 + (i + 1) * (cw + cg), 450, (52, 168, 83))

    # ---------------- 输出层 ----------------
    d.text((120, 570), "③ 输出层", font=f(28, bold=True), fill=C_OUT_BD)
    outs = [
        ("标注画面", ["合规绿框 / 违规红框"]),
        ("实时统计面板", ["人数 / 告警 / FPS"]),
        ("结果保存", ["截图 / 标注视频"]),
        ("可视化界面", ["tkinter，四路切换"]),
    ]
    ow, og = 380, 34
    for i, (t, ls) in enumerate(outs):
        box(d, 120 + i * (ow + og), 615, ow, 130, t, ls, C_OUT, C_OUT_BD, 28, 21)
        arrow(d, 120 + i * (cw + cg) + cw // 2, 528, 120 + i * (ow + og) + ow // 2, 612)

    # ---------------- 支撑层 ----------------
    d.text((120, 790), "④ 支撑层（数据与训练流水线）", font=f(28, bold=True), fill=C_SUP_BD)
    sup = [
        ("数据准备", ["COCO→YOLO 转换", "去重/统一 640/8:1:1"]),
        ("训练与消融", ["YOLOv8n / +SE / +CBAM", "25 轮，同数据同超参"]),
        ("评估", ["mAP / P / R / FPS", "val + test 双集"]),
        ("导出部署", ["ONNX / TorchScript", "GPU / CPU 可切"]),
    ]
    sw, sg = 380, 34
    for i, (t, ls) in enumerate(sup):
        box(d, 120 + i * (sw + sg), 835, sw, 130, t, ls, C_SUP, C_SUP_BD, 28, 21)

    # 支撑层与核心层的关系用底部说明文字表达，不画虚线、不叠加标注（都会压住方框）
    d.text((120, 1000), "说明：①→②→③ 为推理时的数据流；④ 为离线训练阶段，训练产出的权重供 ② 加载。",
           font=f(22), fill=C_GRAY)
    d.text((120, 1040), "输入路径与界面选项一一对应，四个输入源共用同一套检测/跟踪/告警逻辑。",
           font=f(22), fill=C_GRAY)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, quality=95)
    print(f"已生成: {OUT}  ({img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    main()
