"""摄像头定时抓拍 —— 用于"自主采集数据集"（评分项：数据集来源与规模）。

使用方法：
    # 默认摄像头，每 1.5 秒存一张，按 q 退出
    python src/collect_images.py --name helmet_0915

    # 指定间隔与分辨率
    python src/collect_images.py --name site_a --interval 1.0 --width 1280 --height 720

采集建议（详见 docs/01_数据采集与标注规范.md）：
    * 同一场景不要连续拍几百张几乎一样的图，人的角度/距离/遮挡要变化；
    * 按"场景_日期"命名子目录，便于划分数据集时按场景分层，避免训练/验证集同源泄漏；
    * 目标数量目标：train+val ≥ 800 张，其中至少 15% 包含"未戴安全帽"的负样本。
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

from config import RAW_DIR

WINDOW = "collect - press q to quit, s to save current frame"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="摄像头定时抓拍，用于自建 YOLO 数据集")
    p.add_argument("--name", default="session", help="子目录名，建议写成 场景_日期")
    p.add_argument("--camera", type=int, default=0, help="摄像头编号，笔记本内置一般为 0")
    p.add_argument("--interval", type=float, default=1.5, help="自动抓拍间隔（秒）")
    p.add_argument("--max", type=int, default=0, help="最多抓拍张数，0 表示不限制")
    p.add_argument("--width", type=int, default=1280, help="采集宽度")
    p.add_argument("--height", type=int, default=720, help="采集高度")
    p.add_argument("--no-preview", action="store_true", help="不显示预览窗口（无显示器环境）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = RAW_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)  # CAP_DSHOW: Windows 下打开更快
    if not cap.isOpened():
        raise SystemExit(f"无法打开摄像头 {args.camera}，请检查是否被其他程序占用")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    saved = 0
    last_save = 0.0
    print(f"抓拍目录: {out_dir}")
    print(f"间隔 {args.interval}s；按 q 退出，按 s 立即保存当前帧")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("读取画面失败，退出")
                break

            now = time.time()
            auto = (now - last_save) >= args.interval
            if auto or cv2.waitKey(1) & 0xFF == ord("s"):
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                cv2.imwrite(str(out_dir / f"{args.name}_{stamp}.jpg"), frame)
                saved += 1
                last_save = now
                print(f"  [{saved}] {stamp}.jpg")
                if args.max and saved >= args.max:
                    print(f"已达上限 {args.max} 张，停止采集")
                    break

            if not args.no_preview:
                tip = f"saved={saved}  next in {max(0.0, args.interval - (time.time() - last_save)):.1f}s"
                cv2.putText(frame, tip, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow(WINDOW, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.03)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"采集结束，共保存 {saved} 张到 {out_dir}")


if __name__ == "__main__":
    main()
