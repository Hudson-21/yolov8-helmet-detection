"""图形化检测界面（tkinter，零额外依赖）。

对应评分项：部署拓展性 4 分（可拓展简易界面）+ 运行演示效果 6 分。
答辩演示时直接双击 `run_ui.bat` 或在本目录执行 `python src/ui_app.py` 即可。

功能：
    * 三种输入：单张图片 / 视频文件 / 实时摄像头；
    * 阈值可调：置信度、NMS IoU（弱光降 conf 找目标，密集人群降 iou 防吞并）；
    * 设备切换：GPU / CPU（现场没独显的机器也能跑）；
    * 实时统计：已戴安全帽人数、未戴人数、视野内总人数、FPS；
    * 未戴安全帽告警：红色高亮 + 状态灯；
    * 结果保存：当前帧截图、整段视频录像。
"""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from config import DEVICE, PRETRAINED_WEIGHTS, RESULTS_DIR, TRAINED_DIR
from detect import HelmetDetector

CANVAS_W, CANVAS_H = 860, 560
PANEL_W = 320


class HelmetApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("安全帽佩戴检测系统")
        root.geometry(f"{CANVAS_W + PANEL_W + 24}x{CANVAS_H + 40}")
        root.configure(bg="#1e1e1e")
        root.resizable(False, False)

        self.detector: HelmetDetector | None = None
        self.cap: cv2.VideoCapture | None = None
        self.writer: cv2.VideoWriter | None = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.worker: threading.Thread | None = None
        self.running = tk.BooleanVar(value=False)
        self.save_video = tk.BooleanVar(value=False)
        self._photo: ImageTk.PhotoImage | None = None
        self._last_alarm = 0.0

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ----------------------------------------------------------------- UI --
    def _build_ui(self) -> None:
        left = tk.Frame(self.root, bg="#121212")
        left.pack(side=tk.LEFT, padx=8, pady=8)
        self.canvas = tk.Canvas(left, width=CANVAS_W, height=CANVAS_H, bg="#000000", highlightthickness=0)
        self.canvas.pack()
        self.placeholder = self.canvas.create_text(
            CANVAS_W // 2, CANVAS_H // 2, text="选择输入源开始检测", fill="#7a7a7a", font=("Microsoft YaHei", 18)
        )

        right = tk.Frame(self.root, bg="#1e1e1e")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(0, 10), pady=10)

        def section(text: str) -> None:
            tk.Label(right, text=text, bg="#1e1e1e", fg="#8ab4f8", font=("Microsoft YaHei", 11, "bold")).pack(
                anchor="w", pady=(10, 4)
            )

        # ---- 权重 ----
        section("模型权重")
        self.weights_var = tk.StringVar(value=str(self._default_weights()))
        tk.Entry(right, textvariable=self.weights_var, width=34, font=("Consolas", 8)).pack(anchor="w")
        ttk.Button(right, text="选择权重…", command=self.choose_weights).pack(anchor="w", pady=4)

        # ---- 阈值 ----
        section("检测阈值")
        self.conf_var = tk.DoubleVar(value=0.25)
        self.iou_var = tk.DoubleVar(value=0.45)
        self._slider(right, "置信度 conf", self.conf_var, 0.05, 0.9)
        self._slider(right, "重叠 IoU", self.iou_var, 0.1, 0.9)

        # ---- 设备 ----
        section("推理设备")
        self.device_var = tk.StringVar(value=str(DEVICE))
        dev_row = tk.Frame(right, bg="#1e1e1e")
        dev_row.pack(anchor="w")
        ttk.Radiobutton(dev_row, text="GPU", value="0", variable=self.device_var).pack(side=tk.LEFT)
        ttk.Radiobutton(dev_row, text="CPU", value="cpu", variable=self.device_var).pack(side=tk.LEFT, padx=8)

        # ---- 输入 ----
        section("输入源")
        btns = tk.Frame(right, bg="#1e1e1e")
        btns.pack(anchor="w")
        ttk.Button(btns, text="图片", width=8, command=self.open_image).pack(side=tk.LEFT)
        ttk.Button(btns, text="视频", width=8, command=self.open_video).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="摄像头", width=8, command=self.open_camera).pack(side=tk.LEFT)
        ttk.Button(right, text="■ 停止", command=self.stop).pack(anchor="w", pady=6)

        ttk.Checkbutton(right, text="同时录制结果视频", variable=self.save_video).pack(anchor="w")
        ttk.Button(right, text="保存当前帧截图", command=self.save_frame).pack(anchor="w", pady=4)

        # ---- 统计 ----
        section("实时统计")
        self.stats_var = tk.StringVar(value="已戴安全帽 0 人 | 未戴 0 人\n视野内总人数 0 | FPS 0.0")
        tk.Label(
            right, textvariable=self.stats_var, bg="#121212", fg="#e0e0e0", justify="left", font=("Consolas", 10), padx=8, pady=6
        ).pack(anchor="w", fill=tk.X)

        self.alarm_label = tk.Label(
            right, text="● 正常", bg="#121212", fg="#4caf50", font=("Microsoft YaHei", 14, "bold"), pady=8
        )
        self.alarm_label.pack(anchor="w", fill=tk.X, pady=(8, 0))

        self.status = tk.StringVar(value="就绪")
        tk.Label(right, textvariable=self.status, bg="#1e1e1e", fg="#888", font=("Microsoft YaHei", 9)).pack(
            anchor="w", side=tk.BOTTOM, pady=6
        )

    @staticmethod
    def _slider(parent: tk.Widget, label: str, var: tk.DoubleVar, lo: float, hi: float) -> None:
        row = tk.Frame(parent, bg="#1e1e1e")
        row.pack(anchor="w", fill=tk.X, pady=2)
        tk.Label(row, text=label, bg="#1e1e1e", fg="#ccc", width=10, anchor="w").pack(side=tk.LEFT)
        val = tk.Label(row, text=f"{var.get():.2f}", bg="#1e1e1e", fg="#e0e0e0", width=5)
        val.pack(side=tk.RIGHT)
        s = ttk.Scale(row, from_=lo, to=hi, variable=var, orient=tk.HORIZONTAL, length=150)
        s.pack(side=tk.LEFT, padx=4)
        s.configure(command=lambda _v: val.configure(text=f"{var.get():.2f}"))

    @staticmethod
    def _default_weights() -> Path:
        best = TRAINED_DIR / "best.pt"
        return best if best.exists() else PRETRAINED_WEIGHTS

    # ------------------------------------------------------------- 事件处理 --
    def choose_weights(self) -> None:
        p = filedialog.askopenfilename(title="选择权重", filetypes=[("PyTorch 权重", "*.pt")])
        if p:
            self.weights_var.set(p)

    def ensure_detector(self) -> bool:
        if self.detector is not None:
            self.detector.conf = self.conf_var.get()
            self.detector.iou = self.iou_var.get()
            if str(self.detector.device) != self.device_var.get():
                self.detector = None  # 设备变化需重建
        if self.detector is None:
            try:
                self.status.set("加载模型中…")
                self.root.update_idletasks()
                self.detector = HelmetDetector(
                    weights=self.weights_var.get(),
                    conf=self.conf_var.get(),
                    iou=self.iou_var.get(),
                    device=self.device_var.get(),
                )
                self.status.set(f"模型已加载: {Path(self.detector.weights).name}")
            except Exception as exc:  # noqa: BLE001 - 界面层统一兜底提示
                messagebox.showerror("模型加载失败", str(exc))
                self.status.set("模型加载失败")
                return False
        return True

    def open_image(self) -> None:
        p = filedialog.askopenfilename(title="选择图片", filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not p:
            return
        self.stop()
        if not self.ensure_detector():
            return
        frame = cv2.imread(p)
        if frame is None:
            messagebox.showerror("读取失败", "无法解码该图片")
            return
        res = self.detector.analyze(frame)
        self._render(res.annotated)
        self._update_stats(res.counts, res.fps, res.alarm)
        self.status.set(f"图片: {Path(p).name}")

    def open_video(self) -> None:
        p = filedialog.askopenfilename(title="选择视频", filetypes=[("视频", "*.mp4 *.avi *.mov *.mkv")])
        if not p:
            return
        self._start_stream(p)

    def open_camera(self) -> None:
        self._start_stream(0)

    def _start_stream(self, source) -> None:
        self.stop()
        if not self.ensure_detector():
            return
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if isinstance(source, int) else cv2.VideoCapture(str(source))
        if not cap.isOpened():
            messagebox.showerror("打开失败", f"无法打开视频源: {source}")
            return
        self.cap = cap
        self.detector.reset_tracker()
        self.canvas.delete(self.placeholder)

        if self.save_video.get():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            out = RESULTS_DIR / "videos" / f"ui_{time.strftime('%m%d_%H%M%S')}.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            self.status.set(f"录制中 → {out.name}")

        self.running.set(True)
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()
        self.root.after(15, self._drain)

    # ------------------------------------------------------------- 工作线程 --
    def _loop(self) -> None:
        while self.running.get() and self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:
                self.running.set(False)
                break
            try:
                res = self.detector.analyze(frame)
            except Exception as exc:  # noqa: BLE001 - 单帧异常不应中断整段视频
                print(f"[警告] 单帧处理失败: {exc}")
                continue
            if self.writer is not None:
                self.writer.write(res.annotated)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(res)

    def _drain(self) -> None:
        try:
            while True:
                res = self.frame_queue.get_nowait()
                self._render(res.annotated)
                self._update_stats(res.counts, res.fps, res.alarm)
        except queue.Empty:
            pass
        if self.running.get():
            self.root.after(15, self._drain)
        else:
            self.status.set("已停止")

    # ------------------------------------------------------------- 渲染/统计 --
    def _render(self, bgr) -> None:
        self._last_bgr = bgr  # 供“保存当前帧”使用
        h, w = bgr.shape[:2]
        scale = min(CANVAS_W / w, CANVAS_H / h)
        resized = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(CANVAS_W // 2, CANVAS_H // 2, image=self._photo)

    def _update_stats(self, counts: dict, fps: float, alarm: bool) -> None:
        self.stats_var.set(
            f"已戴安全帽 {counts['hardhat']} 人 | 未戴 {counts['no_hardhat']} 人\n"
            f"视野内总人数 {counts['total']} | FPS {fps:.1f}"
        )
        now = time.time()
        if alarm:
            self._last_alarm = now
        fresh = (now - self._last_alarm) < 2.0
        if fresh:
            self.alarm_label.configure(text="● 告警：发现未佩戴安全帽", fg="#ff5252")
        else:
            self.alarm_label.configure(text="● 正常", fg="#4caf50")

    # ---------------------------------------------------------------- 操作 --
    def save_frame(self) -> None:
        if getattr(self, "_last_bgr", None) is None:
            messagebox.showinfo("提示", "当前没有可保存的画面，请先加载图片或视频")
            return
        out = RESULTS_DIR / "images" / f"ui_{time.strftime('%m%d_%H%M%S')}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), self._last_bgr)
        self.status.set(f"截图已保存: {out.name}")

    def stop(self) -> None:
        self.running.set(False)
        if self.worker is not None and self.worker.is_alive():
            self.worker.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def on_close(self) -> None:
        self.stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    HelmetApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
