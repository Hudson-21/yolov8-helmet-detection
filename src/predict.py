"""检测脚本 —— 支持 图片 / 图片文件夹 / 视频文件 / 摄像头 四种输入。

评分项对应：运行演示效果 6 分（图片、视频、实时摄像头都要能跑）。

用法：
    python src/predict.py --source datasets/images/val                 # 整个文件夹
    python src/predict.py --source demo.jpg --show                      # 单张图并弹窗
    python src/predict.py --source demo.mp4                             # 视频，结果存 results/videos
    python src/predict.py --source 0                                    # 摄像头实时检测
    python src/predict.py --source 0 --conf 0.15                        # 弱光/模糊场景降阈值找回目标
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from config import CONF_THRES, DEVICE, IOU_THRES, RESULTS_DIR
from detect import HelmetDetector

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="安全帽检测推理")
    p.add_argument("--source", required=True, help="图片/文件夹/视频路径，或 0 表示摄像头")
    p.add_argument("--weights", default=None, help="权重路径，缺省用 weights/trained/best.pt")
    p.add_argument("--conf", type=float, default=CONF_THRES)
    p.add_argument("--iou", type=float, default=IOU_THRES)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=DEVICE, help="0=GPU，cpu=CPU")
    p.add_argument("--no-track", action="store_true", help="关闭跟踪，只做逐帧检测")
    p.add_argument("--show", action="store_true", help="弹窗实时显示（摄像头/视频必开）")
    p.add_argument("--out", type=Path, default=None, help="输出路径；缺省自动写到 results/")
    return p.parse_args()


def process_image(det: HelmetDetector, path: Path, show: bool, out: Path) -> None:
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"[跳过] 无法读取 {path}")
        return
    res = det.analyze(frame)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), res.annotated)
    print(
        f"{path.name}: 已戴={res.counts['hardhat']} 未戴={res.counts['no_hardhat']} "
        f"总人数={res.counts['total']} 告警={'是' if res.alarm else '否'} → {out}"
    )
    if show:
        cv2.imshow("result", res.annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_stream(det: HelmetDetector, source, show: bool, out: Path | None) -> None:
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if isinstance(source, int) else cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频源: {source}")

    writer = None
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))

    frames = alarms = 0
    t_start = time.perf_counter()
    det.reset_tracker()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = det.analyze(frame)
            frames += 1
            alarms += int(res.alarm)
            if writer is not None:
                writer.write(res.annotated)
            if show:
                cv2.imshow("safety-helmet detection", res.annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    cost = time.perf_counter() - t_start
    print(f"处理 {frames} 帧，耗时 {cost:.1f}s，平均 {frames / max(cost, 1e-6):.1f} FPS，告警触发 {alarms} 次")
    if out is not None:
        print(f"结果视频: {out}")


def main() -> None:
    args = parse_args()
    det = HelmetDetector(weights=args.weights, conf=args.conf, iou=args.iou, device=args.device,
                         use_track=not args.no_track, imgsz=args.imgsz)
    print(f"权重: {det.weights}")

    src = args.source
    stamp = time.strftime("%m%d_%H%M%S")

    if src.isdigit():  # 摄像头
        process_stream(det, int(src), show=True, out=args.out or RESULTS_DIR / "videos" / f"camera_{stamp}.mp4")
        return

    path = Path(src)
    if path.is_dir():
        out_dir = args.out or RESULTS_DIR / "images" / f"batch_{stamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for img in sorted(p for p in path.iterdir() if p.suffix.lower() in IMG_EXTS):
            process_image(det, img, show=False, out=out_dir / img.name)
        print(f"\n批量结果目录: {out_dir}")
    elif path.suffix.lower() in VIDEO_EXTS:
        process_stream(det, path, show=args.show, out=args.out or RESULTS_DIR / "videos" / f"{path.stem}_pred.mp4")
    elif path.suffix.lower() in IMG_EXTS:
        process_image(det, path, show=args.show, out=args.out or RESULTS_DIR / "images" / f"{path.stem}_pred.jpg")
    else:
        raise SystemExit(f"不支持的输入: {path}")


if __name__ == "__main__":
    main()
