# -*- coding: utf-8 -*-
"""
live_dialog.py — 自我演化检测系统 · 实时检测 GUI 窗口

挂到「训练」菜单的 Live Detection。在一个大窗口里实时显示:摄像头 / 视频文件(可多选) / 图片(可多选),
经当前线上模型(后台自动热替换)做 ByteTrack 跟踪,画稳定 ID 的框 + 角标。
推理/读帧在 QThread 后台跑,通过信号把帧送回界面显示,不卡 UI。

跟踪用 ultralytics 内置 ByteTrack(model.track(persist=True)):
- 每个处理帧都跟踪(保证 track ID 稳定),同一目标始终同色,标签 `类名 #tid conf`;
- 卡尔曼运动预测扛快速运动,不会像逐帧 IoU 关联那样在快速移动时乱跳 ID/重影;
- 提速靠两手:① 4K→显示宽度 降采样(主要提速,检测精度几乎不变);
  ② grab 丢帧追帧(按真实时间播,落后就跳帧,不累积延迟)——ByteTrack 看到低帧率流仍连续。
注:ByteTrack 解决"框乱/ID 不稳",不解决"4K 解码慢";解码本身慢时需进一步降分辨率或换源。

cv2 读帧、ultralytics 跟踪在本机运行。draw_hud/FpsMeter/clamp_box 等复用 live_runtime。
"""
import os.path as osp
import threading
import time

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QDialog

from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.style import (
    get_dialog_style,
    get_ok_btn_style,
    get_cancel_btn_style,
)

try:
    from anylabeling.services.auto_training.model_server import ModelServer
    from anylabeling.services.auto_training.live_runtime import (
        draw_hud, hud_lines, FpsMeter, clamp_box, color_for_class, _draw_one,
    )
    _IMPORT_OK, _IMPORT_ERR = True, ""
except Exception as _e:  # noqa: BLE001
    _IMPORT_OK, _IMPORT_ERR = False, str(_e)


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm")


def _expand_image_source(source):
    """把图片来源解析成路径列表。支持:
       - ' | ' 分隔的多个文件; - 单个文件; - 单个目录(取目录下所有图)。"""
    import glob

    if " | " in source:
        return [p for p in (s.strip() for s in source.split("|")) if p]
    if osp.isdir(source):
        paths = []
        for e in IMG_EXTS:
            paths += glob.glob(osp.join(source, "*" + e))
            paths += glob.glob(osp.join(source, "*" + e.upper()))
        return sorted(set(paths))
    return [source] if source else []


def _expand_video_source(source):
    """视频来源解析成列表(支持 ' | ' 多选);摄像头编号原样返回。"""
    if " | " in source:
        return [p for p in (s.strip() for s in source.split("|")) if p]
    return [source] if source else []


def fit_display_size(w, h, max_w):
    """把 (w,h) 等比缩到宽不超过 max_w;已小于则不放大。返回 (新w,新h,缩放比例)。
    用于 4K 帧先降采样再处理/显示,大幅减负(检测仍准,因模型本就 resize 到 imgsz)。"""
    if w <= max_w or max_w <= 0:
        return w, h, 1.0
    scale = max_w / float(w)
    return int(round(w * scale)), int(round(h * scale)), scale


def frames_to_skip(elapsed_s, fps, frames_consumed):
    """实时丢帧追帧:按真实经过时间该播到第几帧,若落后则算出要跳过多少帧,避免延迟累积。
    返回应额外丢弃的帧数(>=0)。fps<=0 时不丢帧。"""
    if fps <= 0:
        return 0
    target = int(elapsed_s * fps)        # 按真实时间,现在本应消费到的帧序号
    behind = target - frames_consumed
    return max(0, behind)


def draw_tracks(img, tracks):
    """画 ByteTrack 跟踪框:按 track id 给每个目标固定颜色(同一个人始终同色),
    标签 `类名 #tid conf`。tid 为 None(未确认)时只画 `类名 conf` 且按类别取色。
    不做透明度渐变——ByteTrack 自身处理目标出现/消失,框稳定不重影。"""
    h, w = img.shape[:2]
    for t in tracks:
        key = t.tid if t.tid is not None else t.cls   # 有 tid 按 tid 取色,否则按类别
        color = color_for_class(key)
        x1, y1, x2, y2 = clamp_box(t.x1, t.y1, t.x2, t.y2, w, h)
        if t.tid is not None:
            label = f"{t.label} #{t.tid} {t.conf:.2f}"
        else:
            label = f"{t.label} {t.conf:.2f}"
        _draw_one(img, x1, y1, x2, y2, color, label)


class LiveWorker(QThread):
    frame_ready = pyqtSignal(object)   # QImage
    status = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, opts, parent=None):
        super().__init__(parent)
        self.opts = opts
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def _sleep_ms(self, ms):
        end = time.time() + ms / 1000.0
        while time.time() < end and not self._stop.is_set():
            time.sleep(0.02)

    def run(self):
        try:
            import cv2
        except ImportError:
            self.failed.emit("需要 opencv-python:pip install opencv-python")
            return
        o = self.opts

        # 加载模型(注册表当前模型 + 后台热替换;或固定 yolo11n.pt 测试)
        try:
            server = ModelServer(o["dataset"] or ".", device=o["device"],
                                 conf=o["conf"], iou=0.45, imgsz=o["imgsz"])
            state = {"banner": None}

            def on_swap(old_id, entry):
                m = (entry.get("metrics", {}) or {}).get("map5095")
                ms = f"{m:.3f}" if isinstance(m, (int, float)) else "-"
                state["banner"] = (f"swapped {old_id} -> {entry['id']}  mAP {ms}", time.time() + 2.5)

            if o["use_yolo11n"]:
                server.load_ckpt("yolo11n.pt")
            else:
                server.on_swap = on_swap
                server.load_current()
                server.start_watch(interval=o["interval"])
        except Exception as ex:  # noqa: BLE001
            self.failed.emit("加载模型失败: %s" % ex)
            return

        fpsm = FpsMeter()
        max_disp_w = o.get("max_disp_w", 1280)   # 显示/处理前把 4K 降到这个宽度,大幅减负

        def render_frame(frame):
            """对一帧:先降采样,再用 ByteTrack 跟踪(每帧都跟,保证 ID 稳定),画框+角标。
            提速靠降采样 + 上层 grab 丢帧;ByteTrack 看到低帧率流仍能靠卡尔曼维持轨迹。"""
            h0, w0 = frame.shape[:2]
            nw, nh, _ = fit_display_size(w0, h0, max_disp_w)
            small = cv2.resize(frame, (nw, nh)) if nw != w0 else frame
            tracks = server.track(small)
            banner = None
            b = state.get("banner")
            if b and time.time() < b[1]:
                banner = b[0]
            lines = hud_lines(server.info(), fpsm.tick(), len(tracks), len(tracks))
            draw_tracks(small, tracks)
            draw_hud(small, lines, banner)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QtGui.QImage(rgb.data, w, h, ch * w,
                                QtGui.QImage.Format.Format_RGB888).copy()
            self.frame_ready.emit(qimg)

        try:
            if o["mode"] == "images":
                paths = _expand_image_source(o["source"])
                if not paths:
                    self.failed.emit("没有找到图片: %s" % o["source"])
                    server.stop()
                    return
                i = 0
                while not self._stop.is_set():
                    frame = cv2.imread(paths[i])
                    if frame is not None:
                        render_frame(frame)   # 图片模式每张都跟踪
                    i = (i + 1) % len(paths)
                    self._sleep_ms(o["dwell_ms"])
            else:
                vids = _expand_video_source(o["source"])
                if not vids:
                    self.failed.emit("没有视频源")
                    server.stop()
                    return
                vi = 0
                while not self._stop.is_set():
                    one = vids[vi]
                    src = int(one) if str(one).isdigit() else one
                    cap = cv2.VideoCapture(src)
                    if not cap.isOpened():
                        self.failed.emit("无法打开视频源: %s" % one)
                        server.stop()
                        return
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    is_camera = str(one).isdigit()
                    t0 = time.time()
                    consumed = 0      # 已消费(读取)的帧数
                    while not self._stop.is_set():
                        ok, frame = cap.read()
                        if not ok:
                            break
                        consumed += 1
                        # 实时丢帧追帧:视频源按真实时间该播到哪,落后就丢中间帧(不累积延迟)
                        if not is_camera:
                            skip = frames_to_skip(time.time() - t0, fps, consumed)
                            if skip > 0:
                                for _ in range(skip):
                                    if not cap.grab():
                                        break
                                    consumed += 1
                        render_frame(frame)   # ByteTrack 每个处理帧都跟踪,保证 ID 稳定
                        if is_camera:
                            self._sleep_ms(max(1, int(1000.0 / max(1.0, fps)) - 5))
                    cap.release()
                    vi += 1
                    if vi >= len(vids):
                        if o["mode"] == "video" and o.get("loop") and len(vids) >= 1:
                            vi = 0
                            t0 = time.time()
                            continue
                        break
        except Exception as ex:  # noqa: BLE001
            self.failed.emit("运行出错: %s" % ex)
        finally:
            server.stop()
        self.status.emit("已停止")


class LiveDetectionDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent_widget = parent
        self.worker = None
        self._setup_ui()
        if not _IMPORT_OK:
            self.status_label.setText("✗ 导入失败: " + _IMPORT_ERR)
            self.start_btn.setEnabled(False)

    def _setup_ui(self):
        self.setWindowTitle(self.tr("Live Detection"))
        self.resize(1320, 900)
        self.showMaximized()
        try:
            self.setStyleSheet(get_dialog_style())
        except Exception:
            pass
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        # 控制行
        ctl = QtWidgets.QHBoxLayout()
        ctl.addWidget(QtWidgets.QLabel(self.tr("来源")))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems([self.tr("摄像头"), self.tr("视频(可多选)"), self.tr("图片(可多选)")])
        ctl.addWidget(self.mode_combo)
        self.src_edit = QtWidgets.QLineEdit("0")
        self.src_edit.setPlaceholderText(self.tr("摄像头编号(0)/ 视频路径 / 图片文件夹"))
        ctl.addWidget(self.src_edit, 1)
        self.browse_btn = QtWidgets.QPushButton(self.tr("浏览"))
        ctl.addWidget(self.browse_btn)
        ctl.addWidget(QtWidgets.QLabel(self.tr("设备")))
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItems(["0", "cpu", self.tr("自动")])   # 默认 GPU(0)
        ctl.addWidget(self.device_combo)
        ctl.addWidget(QtWidgets.QLabel(self.tr("模型")))
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems([self.tr("注册表当前模型"), self.tr("yolo11n.pt(通用测试)")])
        ctl.addWidget(self.model_combo)
        ctl.addWidget(QtWidgets.QLabel("conf"))
        self.conf_spin = QtWidgets.QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        ctl.addWidget(self.conf_spin)
        # 实时性参数:把高分辨率视频降到此宽度再处理/显示(主要提速手段)
        ctl.addWidget(QtWidgets.QLabel(self.tr("显示宽度")))
        self.dispw_combo = QtWidgets.QComboBox()
        self.dispw_combo.addItems(["960", "1280", "1920", self.tr("原始")])
        self.dispw_combo.setCurrentText("1280")
        self.dispw_combo.setToolTip(self.tr("把高分辨率视频(如4K)降到此宽度再处理/显示,大幅提速;检测精度几乎不变"))
        ctl.addWidget(self.dispw_combo)
        root.addLayout(ctl)

        # 视频显示区(大、自适应)
        self.video = QtWidgets.QLabel(self.tr("选择来源后点『开始』"))
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(1000, 640)
        self.video.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                 QtWidgets.QSizePolicy.Policy.Expanding)
        self.video.setStyleSheet(
            "background:#0b0d11;border:1px solid #2a2f3a;border-radius:8px;color:#7a8290;"
        )
        root.addWidget(self.video, 1)

        # 底部:开始/停止 + 状态 + 关闭
        bot = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton(self.tr("开始"))
        self.stop_btn = QtWidgets.QPushButton(self.tr("停止"))
        self.stop_btn.setEnabled(False)
        self.status_label = QtWidgets.QLabel("")
        bot.addWidget(self.start_btn)
        bot.addWidget(self.stop_btn)
        bot.addWidget(self.status_label, 1)
        self.close_btn = QtWidgets.QPushButton(self.tr("关闭"))
        bot.addWidget(self.close_btn)
        try:
            self.start_btn.setStyleSheet(get_ok_btn_style())
            self.close_btn.setStyleSheet(get_cancel_btn_style())
        except Exception:
            pass
        root.addLayout(bot)

        self.browse_btn.clicked.connect(self._browse)
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        self.close_btn.clicked.connect(self.reject)
        self._mode_changed()

    # ------------------------------------------------------------------
    def _mode_changed(self):
        m = self.mode_combo.currentIndex()
        self.browse_btn.setEnabled(m != 0)
        if m == 0:
            self.src_edit.setText("0")
        elif not self.src_edit.text() or self.src_edit.text() == "0":
            self.src_edit.setText("")

    def _browse(self):
        m = self.mode_combo.currentIndex()
        if m == 1:
            # 视频:可多选(批量依次处理);也可只选一个
            fs, _ = QtWidgets.QFileDialog.getOpenFileNames(
                self, self.tr("选择视频(可多选)"), "",
                "视频 (*.mp4 *.avi *.mov *.mkv *.flv);;所有文件 (*.*)")
            if fs:
                self.src_edit.setText(" | ".join(fs))
        elif m == 2:
            # 图片:可多选单张/多张;也可点"选文件夹"按钮选整个目录
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle(self.tr("选择图片"))
            box.setText(self.tr("选择单张/多张图片,还是整个文件夹?"))
            b_files = box.addButton(self.tr("选图片(可多选)"), QtWidgets.QMessageBox.ButtonRole.AcceptRole)
            b_dir = box.addButton(self.tr("选文件夹"), QtWidgets.QMessageBox.ButtonRole.AcceptRole)
            box.addButton(self.tr("取消"), QtWidgets.QMessageBox.ButtonRole.RejectRole)
            box.exec()
            if box.clickedButton() is b_files:
                fs, _ = QtWidgets.QFileDialog.getOpenFileNames(
                    self, self.tr("选择图片(可多选)"), "",
                    "图片 (*.jpg *.jpeg *.png *.bmp *.webp);;所有文件 (*.*)")
                if fs:
                    self.src_edit.setText(" | ".join(fs))
            elif box.clickedButton() is b_dir:
                d = QtWidgets.QFileDialog.getExistingDirectory(self, self.tr("选择图片文件夹"))
                if d:
                    self.src_edit.setText(d)

    def _dataset(self):
        st = getattr(self.parent_widget, "settings", None)
        if st is not None:
            v = st.value("publish/output_dir", "")
            if v:
                return str(v)
        return ""

    def _start(self):
        if self.worker is not None:
            return
        m = self.mode_combo.currentIndex()
        mode = {0: "camera", 1: "video", 2: "images"}[m]
        dev = self.device_combo.currentText()
        dev = None if dev == self.tr("自动") else dev
        src = self.src_edit.text().strip()
        if mode != "camera" and not src:
            self.status_label.setText(self.tr("请先选择视频/图片来源"))
            return
        dispw_txt = self.dispw_combo.currentText()
        max_disp_w = 0 if dispw_txt == self.tr("原始") else int(dispw_txt)
        opts = dict(
            dataset=self._dataset(), device=dev, conf=self.conf_spin.value(), imgsz=640,
            smooth=0.4, interval=2.0, mode=mode, source=src,
            use_yolo11n=(self.model_combo.currentIndex() == 1), dwell_ms=1200, loop=True,
            max_disp_w=max_disp_w,
        )
        self.worker = LiveWorker(opts)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.failed.connect(self._on_failed)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished.connect(self._on_worker_finished)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText(self.tr("运行中…"))
        self.worker.start()

    def _on_frame(self, qimg):
        pm = QtGui.QPixmap.fromImage(qimg)
        self.video.setPixmap(pm.scaled(
            self.video.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def _on_failed(self, msg):
        self.status_label.setText("✗ " + msg)
        self._cleanup_worker()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _stop(self):
        if self.worker:
            self.worker.stop()
        self.stop_btn.setEnabled(False)
        self.status_label.setText(self.tr("正在停止…"))

    def _on_worker_finished(self):
        self._cleanup_worker()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _cleanup_worker(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(3000)
            self.worker = None

    def reject(self):
        self._cleanup_worker()
        super().reject()

    def closeEvent(self, ev):
        # 非模态 + WA_DeleteOnClose 时,点窗口X走这里,确保后台线程被停掉
        self._cleanup_worker()
        super().closeEvent(ev)
