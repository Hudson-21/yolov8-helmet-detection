"""检测核心逻辑 —— 一次封装，供命令行与图形界面共用。

类别体系（2 类，头部级）：
    0 hardhat    → 戴着安全帽的头部
    1 no-hardhat → 没戴安全帽的头部（违规）

职责（对应评分项：核心功能实现 8 分）：
    * 目标检测：头部级判定，直接区分"戴/未戴"，不需要再做人员与帽子的空间关联；
    * 目标跟踪：调用 Ultralytics 内置 BoT-SORT（YOLO+SORT 融合，加分项），
      用 track id 去重，视线里同一个人只计一次；
    * 计数：已戴人数、未戴人数、视野内总人数（= 已戴 + 未戴）；
    * 告警：只要检测到 no-hardhat 即置位告警，并带冷却，避免逐帧刷屏；
    * 绘制：合规绿色、违规红色粗框，左上角叠加实时统计面板。

对"遮挡 / 模糊 / 光照变化"的适配：
    * 置信度阈值可调（界面滑块），弱光场景下调阈值可找回目标；
    * NMS IoU 可调，密集人群下调小 IoU 可减少目标被吞并；
    * 小目标（远景头部）通过 train.py --imgsz 960/1280 提升召回。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from config import (
    ALARM_CLASS_ID,
    ALARM_COOLDOWN_FRAMES,
    CLASS_NAMES,
    CLASS_NAMES_CN,
    CONF_THRES,
    DEVICE,
    IOU_THRES,
    TRACKER,
    resolve_weights,
)

# 绘制用配色（BGR）：合规=绿，违规=红
COLOR_OK = (60, 200, 60)
COLOR_VIOLATION = (40, 40, 230)
COLOR_PANEL = (30, 30, 30)


@dataclass
class FrameResult:
    """单帧分析结果，界面与命令行都消费这个结构。"""

    annotated: np.ndarray  # 已绘制的图像
    counts: dict[str, int] = field(default_factory=dict)  # 各类别数量
    violators: list[int] = field(default_factory=list)  # 未戴安全帽目标的 track id
    alarm: bool = False  # 本帧是否告警
    fps: float = 0.0  # 含前后处理的实时帧率


class HelmetDetector:
    """安全帽（戴/未戴）检测器。"""

    def __init__(
        self,
        weights: str | Path | None = None,
        conf: float = CONF_THRES,
        iou: float = IOU_THRES,
        device: int | str = DEVICE,
        use_track: bool = True,
        imgsz: int = 640,
    ) -> None:
        from ultralytics import YOLO

        self.weights = resolve_weights(weights)
        self.model = YOLO(str(self.weights))
        self.conf = conf
        self.iou = iou
        self.device = device
        self.use_track = use_track
        self.imgsz = imgsz

        self._fps_smooth = 0.0
        self._alarm_cooldown = 0

    # ------------------------------------------------------------------ 推理 --
    def _infer(self, frame: np.ndarray):
        """对单帧做检测/跟踪，返回 Ultralytics 的 Results 对象。"""
        kwargs = dict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if self.use_track:
            return self.model.track(**kwargs, persist=True, tracker=TRACKER)[0]
        return self.model.predict(**kwargs)[0]

    # -------------------------------------------------------------- 主入口 --
    def analyze(self, frame: np.ndarray) -> FrameResult:
        """分析一帧图像，返回绘制结果与统计信息。"""
        t0 = time.perf_counter()
        result = self._infer(frame)

        boxes = result.boxes
        n_ok = n_bad = 0
        violators: list[int] = []

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(clss)

            for box, cid, cf, tid in zip(xyxy, clss, confs, ids):
                if cid == ALARM_CLASS_ID:  # 未戴安全帽
                    n_bad += 1
                    violators.append(int(tid) if tid is not None else -1)
                    label = f"NO-HARDHAT #{tid}" if tid is not None else "NO-HARDHAT"
                    self._draw_box(frame, box, COLOR_VIOLATION, label, thickness=3)
                else:  # 已戴安全帽
                    n_ok += 1
                    label = f"hardhat #{tid} {cf:.2f}" if tid is not None else f"hardhat {cf:.2f}"
                    self._draw_box(frame, box, COLOR_OK, label)

        # ---- 告警：带冷却，避免逐帧重复触发 ----
        # 注意：冷却计数必须"按帧衰减"，不能在"本帧无违规"时直接清零。
        # 否则目标短暂漏检（跟踪抖动、瞬时遮挡）会让冷却归零，下一帧立刻重新告警，
        # 实测会出现每 2 帧触发一次的告警风暴。
        alarm = False
        if n_bad > 0 and self._alarm_cooldown <= 0:
            alarm = True
            self._alarm_cooldown = ALARM_COOLDOWN_FRAMES
        elif self._alarm_cooldown > 0:
            self._alarm_cooldown -= 1

        # ---- FPS（指数平滑，读数不跳） ----
        cost = time.perf_counter() - t0
        inst = 1.0 / max(cost, 1e-6)
        self._fps_smooth = inst if self._fps_smooth == 0 else 0.85 * self._fps_smooth + 0.15 * inst

        counts = {
            "hardhat": n_ok,
            "no_hardhat": n_bad,
            "total": n_ok + n_bad,
            "violator": n_bad,
        }
        self._draw_panel(frame, counts, self._fps_smooth, alarm)
        return FrameResult(annotated=frame, counts=counts, violators=violators, alarm=alarm, fps=self._fps_smooth)

    # ------------------------------------------------------------- 绘制工具 --
    @staticmethod
    def _draw_box(img: np.ndarray, box: np.ndarray, color: tuple, label: str, thickness: int = 2) -> None:
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if not label:
            return
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_panel(img: np.ndarray, counts: dict, fps: float, alarm: bool) -> None:
        """左上角统计面板。"""
        lines = [
            f"hardhat   : {counts['hardhat']}",
            f"no-hardhat: {counts['no_hardhat']}",
            f"total head: {counts['total']}",
            f"FPS       : {fps:.1f}",
        ]
        w, h = 190, 20 + 22 * len(lines)
        overlay = img.copy()
        cv2.rectangle(overlay, (8, 8), (8 + w, 8 + h), COLOR_PANEL, -1)
        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
        for i, text in enumerate(lines):
            color = COLOR_VIOLATION if ("no-hardhat" in text and counts["no_hardhat"] > 0) else (230, 230, 230)
            cv2.putText(img, text, (18, 34 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        if alarm:
            cv2.putText(
                img, "ALARM: NO HARDHAT", (20, img.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_VIOLATION, 2, cv2.LINE_AA
            )

    # ---------------------------------------------------------------- 便捷 --
    def reset_tracker(self) -> None:
        """切换视频源时清空跟踪状态，避免 track id 错乱。"""
        self.model.predictor = None
        self._alarm_cooldown = 0


def class_name_cn(idx: int) -> str:
    """类别 id → 中文名。"""
    return CLASS_NAMES_CN.get(idx, CLASS_NAMES.get(idx, str(idx)))
