# -*- coding: utf-8 -*-
"""
collect_dialog.py — 自我演化系统 · 数据采集窗口(闭环的"持续采集"一环)

从视频(可多选)按策略采集帧 -> 存按时间命名的会话文件夹 -> 可一键用 current 模型自动预标注
-> 进入步骤一(SelfEvolvingRecognition 里筛选/修正)-> 选择性发布 -> 训练 ... 形成自进化闭环。

策略:变化触发(帧差) + 可选检测触发(当前模型检到目标才采) + min_gap 去抖 + max_gap 兜底。
边采边显示:画面实时显示,检测开时画框。
"""
import os
import os.path as osp
import threading
import time

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QDialog

from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.style import (
    get_dialog_style, get_ok_btn_style, get_cancel_btn_style,
)

try:
    from anylabeling.services.auto_training import collector as COL
    from anylabeling.services.auto_training.model_server import ModelServer
    _IMPORT_OK, _IMPORT_ERR = True, ""
except Exception as _e:  # noqa: BLE001
    _IMPORT_OK, _IMPORT_ERR = False, str(_e)


class CollectWorker(QThread):
    frame_ready = pyqtSignal(object)     # QImage(预览)
    stat = pyqtSignal(int, int)          # (已处理帧, 已采集帧)
    log = pyqtSignal(str)
    done = pyqtSignal(str)               # 采集会话目录
    failed = pyqtSignal(str)

    def __init__(self, opts, parent=None):
        super().__init__(parent)
        self.opts = opts
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.failed.emit("需要 opencv-python")
            return
        o = self.opts

        # 可选:仅当勾选「检测触发」时加载当前模型,用来决定"这一帧值不值得采"
        # (不再画框 —— 采集只需看清画面;去框更干净,也省掉平滑器开销)
        server = None
        if o["use_detection"]:
            try:
                server = ModelServer(o["dataset"] or ".", device=o["device"],
                                     conf=o["conf"], imgsz=640)
                server.load_current()
            except Exception as ex:  # noqa: BLE001
                self.failed.emit("检测触发需要当前模型,但加载失败: %s" % ex)
                return

        vids = [p.strip() for p in o["sources"] if p.strip()]
        if not vids:
            self.failed.emit("没有视频源")
            return

        sess = COL.CaptureSession(
            o["out_root"], source_desc=" | ".join(vids),
            strategy={
                "change_thresh": o["change_thresh"], "use_detection": o["use_detection"],
                "min_gap": o["min_gap"], "max_gap": o["max_gap"], "conf": o["conf"],
            },
        )
        self.log.emit("采集会话: %s" % sess.dir)

        prev_gray = None
        last_save_t = -1e9
        processed = 0
        try:
            for vid in vids:
                if self._stop.is_set():
                    break
                src = int(vid) if str(vid).isdigit() else vid
                cap = cv2.VideoCapture(src)
                if not cap.isOpened():
                    self.log.emit("打不开,跳过: %s" % vid)
                    continue
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                infer_every = max(1, int(o.get("infer_every", 3)))
                max_w = o.get("max_proc_w", 1280)
                while not self._stop.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    processed += 1
                    now = processed / max(1.0, fps)   # 用帧号/fps 当"视频内时间",稳定可控

                    # 降采样小图:推理 + 变化检测都在小图上算(4K 直接推理太慢);存帧仍存原始大图
                    h0, w0 = frame.shape[:2]
                    if max_w and w0 > max_w:
                        sc = max_w / float(w0)
                        small = cv2.resize(frame, (int(w0 * sc), int(h0 * sc)))
                    else:
                        small = frame

                    # 检测(可选 + 降频:每 infer_every 帧推一次)
                    dets = []
                    has_det = False
                    do_infer = server is not None and ((processed - 1) % infer_every == 0)
                    if do_infer:
                        try:
                            dets = server.infer(small)
                            has_det = len(dets) > 0
                        except Exception:
                            dets = []
                    # 变化分(在小图灰度上算)
                    cur_gray = cv2.resize(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (64, 64))
                    diff = COL.frame_diff_score(prev_gray, cur_gray)

                    save, reason = COL.should_capture(
                        diff, has_det,
                        change_thresh=o["change_thresh"], use_detection=o["use_detection"],
                        secs_since_last=now - last_save_t,
                        min_gap=o["min_gap"], max_gap=o["max_gap"],
                    )
                    if save:
                        sess.save_frame(frame, reason=reason)   # 存原始大图(训练要高质量)
                        last_save_t = now
                        prev_gray = cur_gray
                        self.log.emit("采集 #%d (%s) diff=%.3f" % (sess.count, reason, diff))
                    # 预览(降频,只显示画面,不画检测框 —— 更干净,也更省)
                    if processed % o["preview_every"] == 0:
                        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgb.shape
                        qimg = QtGui.QImage(rgb.data, w, h, ch * w,
                                            QtGui.QImage.Format.Format_RGB888).copy()
                        self.frame_ready.emit(qimg)
                    self.stat.emit(processed, sess.count)
                cap.release()
        except Exception as ex:  # noqa: BLE001
            self.failed.emit("采集出错: %s" % ex)
            if server:
                server.stop()
            return
        if server:
            server.stop()
        d = sess.finalize()
        self.done.emit(d)


class CollectDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent_widget = parent
        self.worker = None
        self.last_session_dir = ""
        self._open_on_close = ""   # 关窗口后让主界面打开的文件夹(自动标注成功后设置)
        self._setup_ui()
        if not _IMPORT_OK:
            self.status.setText("✗ 导入失败: " + _IMPORT_ERR)
            self.start_btn.setEnabled(False)

    def _setup_ui(self):
        self.setWindowTitle(self.tr("Data Collection"))
        self.resize(1280, 860)
        self.showMaximized()
        try:
            self.setStyleSheet(get_dialog_style())
        except Exception:
            pass
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        # 源
        srow = QtWidgets.QHBoxLayout()
        srow.addWidget(QtWidgets.QLabel(self.tr("视频源(可多选)")))
        self.src_edit = QtWidgets.QLineEdit()
        self.src_edit.setPlaceholderText(self.tr("选择一个或多个视频文件"))
        srow.addWidget(self.src_edit, 1)
        self.browse_btn = QtWidgets.QPushButton(self.tr("浏览"))
        srow.addWidget(self.browse_btn)
        root.addLayout(srow)

        # 采集根目录
        orow = QtWidgets.QHBoxLayout()
        orow.addWidget(QtWidgets.QLabel(self.tr("采集到")))
        self.out_edit = QtWidgets.QLineEdit()
        self.out_edit.setPlaceholderText(self.tr("采集根目录(会在其下按时间建会话文件夹)"))
        orow.addWidget(self.out_edit, 1)
        self.out_browse_btn = QtWidgets.QPushButton(self.tr("浏览"))
        orow.addWidget(self.out_browse_btn)
        root.addLayout(orow)

        # 策略参数
        prow = QtWidgets.QHBoxLayout()
        self.detect_chk = QtWidgets.QCheckBox(self.tr("检测触发(用当前模型)"))
        self.detect_chk.setToolTip(self.tr(
            "不勾(默认):纯按画面变化采帧,不加载模型,最快;"
            "勾上:只在当前模型检到目标时才采(会加载模型推理,较慢)。预览均不画框。"))
        self.detect_chk.setChecked(False)
        prow.addWidget(self.detect_chk)
        prow.addWidget(QtWidgets.QLabel(self.tr("变化阈值")))
        self.change_spin = QtWidgets.QDoubleSpinBox()
        self.change_spin.setRange(0.0, 1.0)
        self.change_spin.setSingleStep(0.01)
        self.change_spin.setValue(0.04)
        self.change_spin.setToolTip(self.tr("帧差≥此值算'有变化'(越小越敏感,越易采)"))
        prow.addWidget(self.change_spin)
        prow.addWidget(QtWidgets.QLabel(self.tr("最短间隔(秒)")))
        self.mingap_spin = QtWidgets.QDoubleSpinBox()
        self.mingap_spin.setRange(0.0, 60.0)
        self.mingap_spin.setSingleStep(0.5)
        self.mingap_spin.setValue(1.0)
        prow.addWidget(self.mingap_spin)
        prow.addWidget(QtWidgets.QLabel(self.tr("兜底间隔(秒)")))
        self.maxgap_spin = QtWidgets.QDoubleSpinBox()
        self.maxgap_spin.setRange(0.0, 3600.0)
        self.maxgap_spin.setSingleStep(10.0)
        self.maxgap_spin.setValue(60.0)
        self.maxgap_spin.setToolTip(self.tr("就算没变化,超过此间隔也强制采一张(0=关)"))
        prow.addWidget(self.maxgap_spin)
        prow.addWidget(QtWidgets.QLabel("conf"))
        self.conf_spin = QtWidgets.QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        prow.addWidget(self.conf_spin)
        prow.addWidget(QtWidgets.QLabel(self.tr("设备")))
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItems(["0", "cpu", self.tr("自动")])   # 默认 GPU(0)
        prow.addWidget(self.device_combo)
        prow.addWidget(QtWidgets.QLabel(self.tr("每N帧检测")))
        self.infer_every_spin = QtWidgets.QSpinBox()
        self.infer_every_spin.setRange(1, 10)
        self.infer_every_spin.setValue(3)
        self.infer_every_spin.setToolTip(self.tr("每隔N帧推理一次(影响检测触发频率);越大越快"))
        prow.addWidget(self.infer_every_spin)
        prow.addWidget(QtWidgets.QLabel(self.tr("处理宽度")))
        self.procw_combo = QtWidgets.QComboBox()
        self.procw_combo.addItems(["960", "1280", "1920", self.tr("原始")])
        self.procw_combo.setCurrentText("1280")
        self.procw_combo.setToolTip(self.tr("推理/变化检测用的降采样宽度(存帧仍存原始大图);4K建议1280"))
        prow.addWidget(self.procw_combo)
        prow.addStretch(1)
        root.addLayout(prow)

        # 预览
        self.video = QtWidgets.QLabel(self.tr("选择视频后点『开始采集』"))
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(900, 520)
        self.video.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                 QtWidgets.QSizePolicy.Policy.Expanding)
        self.video.setStyleSheet(
            "background:#0b0d11;border:1px solid #2a2f3a;border-radius:8px;color:#7a8290;")
        root.addWidget(self.video, 1)

        # 日志
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
        self.log.setStyleSheet(
            "QPlainTextEdit{background:#0f1115;color:#cdd3de;border:1px solid #2a2f3a;"
            "border-radius:8px;padding:6px;}")
        root.addWidget(self.log)

        # 底部
        bot = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton(self.tr("开始采集"))
        self.stop_btn = QtWidgets.QPushButton(self.tr("停止"))
        self.stop_btn.setEnabled(False)
        self.autolabel_btn = QtWidgets.QPushButton(self.tr("把刚采集的送去自动标注"))
        self.autolabel_btn.setEnabled(False)
        self.autolabel_btn.setToolTip(self.tr("用当前模型给刚采集会话的图预标注成本程序的标注 json"))
        self.status = QtWidgets.QLabel("")
        bot.addWidget(self.start_btn)
        bot.addWidget(self.stop_btn)
        bot.addWidget(self.autolabel_btn)
        bot.addWidget(self.status, 1)
        self.close_btn = QtWidgets.QPushButton(self.tr("关闭"))
        bot.addWidget(self.close_btn)
        try:
            self.start_btn.setStyleSheet(get_ok_btn_style())
            self.close_btn.setStyleSheet(get_cancel_btn_style())
        except Exception:
            pass
        root.addLayout(bot)

        self.browse_btn.clicked.connect(self._browse_src)
        self.out_browse_btn.clicked.connect(self._browse_out)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        self.autolabel_btn.clicked.connect(self._autolabel)
        self.close_btn.clicked.connect(self.reject)

    # ------------------------------------------------------------------
    def _browse_src(self):
        fs, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, self.tr("选择视频(可多选)"), "",
            "视频 (*.mp4 *.avi *.mov *.mkv *.flv);;所有文件 (*.*)")
        if fs:
            self.src_edit.setText(" | ".join(fs))

    def _browse_out(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, self.tr("选择采集根目录"))
        if d:
            self.out_edit.setText(d)

    def _dataset(self):
        st = getattr(self.parent_widget, "settings", None)
        if st is not None:
            v = st.value("publish/output_dir", "")
            if v:
                return str(v)
        return ""

    def _device(self):
        d = self.device_combo.currentText()
        return None if d == self.tr("自动") else d

    def _log(self, msg):
        self.log.appendPlainText("• " + msg)
        self.log.ensureCursorVisible()

    def _start(self):
        if self.worker is not None:
            return
        srcs = [s.strip() for s in self.src_edit.text().split("|") if s.strip()]
        if not srcs:
            self.status.setText(self.tr("请先选择视频源"))
            return
        out_root = self.out_edit.text().strip()
        if not out_root:
            self.status.setText(self.tr("请先选择采集根目录"))
            return
        opts = dict(
            sources=srcs, out_root=out_root, dataset=self._dataset(),
            device=self._device(), conf=self.conf_spin.value(),
            use_detection=self.detect_chk.isChecked(),
            change_thresh=self.change_spin.value(),
            min_gap=self.mingap_spin.value(), max_gap=self.maxgap_spin.value(),
            preview_every=3,
            infer_every=self.infer_every_spin.value(),
            max_proc_w=(0 if self.procw_combo.currentText() == self.tr("原始")
                        else int(self.procw_combo.currentText())),
        )
        self.worker = CollectWorker(opts)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.stat.connect(self._on_stat)
        self.worker.log.connect(self._log)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_finished)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.autolabel_btn.setEnabled(False)
        self.status.setText(self.tr("采集中…"))
        self.worker.start()

    def _on_frame(self, qimg):
        pm = QtGui.QPixmap.fromImage(qimg)
        self.video.setPixmap(pm.scaled(
            self.video.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def _on_stat(self, processed, captured):
        self.status.setText(self.tr("已处理 %d 帧 / 采集 %d 张") % (processed, captured))

    def _on_done(self, session_dir):
        self.last_session_dir = session_dir
        self._log(self.tr("采集完成 -> ") + session_dir)
        self.autolabel_btn.setEnabled(bool(session_dir) and _IMPORT_OK)

    def _on_failed(self, msg):
        self.status.setText("✗ " + msg)
        self._cleanup()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _stop(self):
        if self.worker:
            self.worker.stop()
        self.stop_btn.setEnabled(False)
        self.status.setText(self.tr("正在停止…"))

    def _on_finished(self):
        self._cleanup()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _cleanup(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(3000)
            self.worker = None

    def _autolabel(self):
        """用 current 模型给刚采集会话的图预标注成 SelfEvolvingRecognition json。"""
        d = self.last_session_dir
        if not d or not osp.isdir(d):
            self.status.setText(self.tr("没有可标注的采集会话"))
            return
        ds = self._dataset()
        try:
            from ultralytics import YOLO
            from anylabeling.services.auto_training.model_registry import ModelRegistry
            from anylabeling.services.auto_training import collector as COL
        except Exception as ex:  # noqa: BLE001
            self.status.setText("✗ " + str(ex))
            return
        reg = ModelRegistry(ds) if ds else None
        ckpt = reg.current_ckpt() if reg else None
        if not ckpt:
            self.status.setText(self.tr("注册表没有当前模型,无法自动标注"))
            return
        names = None
        cls_txt = osp.join(ds, "classes.txt") if ds else ""
        if cls_txt and osp.exists(cls_txt):
            names = [l.strip() for l in open(cls_txt, encoding="utf-8") if l.strip()]

        self.autolabel_btn.setEnabled(False)
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(Qt.CursorShape.WaitCursor))
        self._log(self.tr("自动标注中(模型: %s)…") % osp.basename(ckpt))
        try:
            model = YOLO(ckpt)

            def predictor(path):
                r = model.predict(path, conf=self.conf_spin.value(),
                                  device=self._device(), verbose=False)[0]
                h, w = r.orig_shape
                dets = []
                b = getattr(r, "boxes", None)
                if b is not None:
                    import numpy as np
                    xyxy = b.xyxy.cpu().numpy() if hasattr(b.xyxy, "cpu") else np.asarray(b.xyxy)
                    cls = b.cls.cpu().numpy() if hasattr(b.cls, "cpu") else np.asarray(b.cls)
                    cf = b.conf.cpu().numpy() if hasattr(b.conf, "cpu") else np.asarray(b.conf)
                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = (float(v) for v in xyxy[i][:4])
                        dets.append((x1, y1, x2, y2, float(cf[i]), int(cls[i])))
                return w, h, dets

            n_img, n_box = COL.autolabel_dir(
                d, predictor, names or {}, conf=self.conf_spin.value())
            self._log(self.tr("自动标注完成: %d 张图, %d 个框 -> 已写 json 到会话目录") % (n_img, n_box))
            self._open_on_close = d   # 关闭本窗口后,主界面自动打开这个文件夹
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle(self.tr("自动标注完成"))
            box.setText(
                self.tr("已为 %d 张图生成标注(%d 框)。\n\n要在主界面打开该文件夹筛选/修正吗?\n%s")
                % (n_img, n_box, d))
            b_now = box.addButton(self.tr("立即打开(并关闭本窗口)"), QtWidgets.QMessageBox.ButtonRole.AcceptRole)
            box.addButton(self.tr("关闭本窗口后打开"), QtWidgets.QMessageBox.ButtonRole.AcceptRole)
            box.exec()
            if box.clickedButton() is b_now:
                self.close()   # 触发 closeEvent -> 主界面打开文件夹
        except Exception as ex:  # noqa: BLE001
            logger.error(f"autolabel failed: {ex}")
            self.status.setText("✗ " + str(ex))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.autolabel_btn.setEnabled(True)

    def _open_folder_in_main(self):
        """让主界面打开自动标注好的会话文件夹(只触发一次)。"""
        d = self._open_on_close
        self._open_on_close = ""
        if not d or not osp.isdir(d):
            return
        pw = self.parent_widget
        if pw is not None and hasattr(pw, "import_image_folder"):
            try:
                pw.import_image_folder(d)
            except Exception as ex:  # noqa: BLE001
                logger.error(f"open collected folder failed: {ex}")

    def reject(self):
        # reject 会触发 closeEvent,真正的清理与"打开文件夹"都在 closeEvent 里做,避免重复
        super().reject()

    def closeEvent(self, ev):
        self._cleanup()
        pending = self._open_on_close
        super().closeEvent(ev)
        # 窗口关闭后,若有刚标注好的会话目录,让主界面打开它
        if pending:
            self._open_folder_in_main()
