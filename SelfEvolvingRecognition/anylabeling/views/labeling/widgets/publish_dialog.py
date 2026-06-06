# -*- coding: utf-8 -*-
"""
publish_dialog.py — 自我演化检测系统 · 第 1 步的 GUI 入口

放在 SelfEvolvingRecognition 的 anylabeling/views/labeling/widgets/ 下。
打开后:扫描当前文件夹的标注 -> 三档分诊表(可勾选发布) -> 导出 YOLO 数据集。
全部逻辑复用 anylabeling.services.auto_training.publish_dataset(与命令行同一套)。
"""
import os.path as osp

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog

from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.style import (
    get_dialog_style,
    get_ok_btn_style,
    get_cancel_btn_style,
)

# 库接口:scan/export 逻辑唯一来源。导入失败时给清晰提示而不是崩溃。
try:
    from anylabeling.services.auto_training.publish_dataset import (
        QualityConfig,
        scan_paths,
        export_records,
    )
    _PUBLISH_OK = True
    _PUBLISH_ERR = ""
except Exception as _e:  # noqa: BLE001
    _PUBLISH_OK = False
    _PUBLISH_ERR = str(_e)


# 三档分诊的行底色(半透明,深浅模式下都还能看清文字)
_VERDICT_BG = {
    "keep": QtGui.QColor(60, 160, 90, 55),
    "review": QtGui.QColor(225, 165, 45, 55),
    "reject": QtGui.QColor(140, 140, 140, 45),
}


class PublishDatasetDialog(QDialog):
    """质量过滤 -> 选择性发布 -> 导出 YOLO。"""

    COLS = ["发布", "图片", "框数", "保留", "最低分", "分诊"]
    # 输出目录持久化到 QSettings 的键;第 2 步训练读同一个键自动定位数据集
    SETTINGS_OUTPUT_KEY = "publish/output_dir"

    def __init__(self, parent):
        super().__init__(parent)
        self.parent_widget = parent
        self.records = []  # 当前扫描出的 ImageRecord,与表格行一一对应
        self._setup_ui()
        if not _PUBLISH_OK:
            self._log(self.tr("无法导入 publish_dataset: ") + _PUBLISH_ERR, error=True)
            self.rescan_btn.setEnabled(False)
            self.export_btn.setEnabled(False)
            return
        self.src_edit.setText(self._default_source_dir())
        self._restore_output_dir()
        self._rescan()  # 打开时按"标注目录"扫描,不再弹框

    # ------------------------------------------------------------------ UI
    def _setup_ui(self):
        self.setWindowTitle(self.tr("Publish Dataset"))
        self.resize(880, 600)
        try:
            self.setStyleSheet(get_dialog_style())
        except Exception:  # 样式拿不到也不影响功能
            pass

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        # 标注目录(源)+ 选择
        src_layout = QtWidgets.QHBoxLayout()
        src_layout.addWidget(QtWidgets.QLabel(self.tr("标注目录")))
        self.src_edit = QtWidgets.QLineEdit()
        self.src_edit.setPlaceholderText(
            self.tr("自动标注生成的 .json 所在目录(一般就是图片目录)")
        )
        src_layout.addWidget(self.src_edit, 1)
        self.choose_src_btn = QtWidgets.QPushButton(self.tr("选择目录"))
        src_layout.addWidget(self.choose_src_btn)
        root.addLayout(src_layout)

        # 阈值 + 重新扫描 + 汇总
        thr = QtWidgets.QHBoxLayout()
        thr.addWidget(QtWidgets.QLabel(self.tr("保留阈值")))
        self.keep_spin = self._spin(0.0, 1.0, 0.05, 0.50)
        thr.addWidget(self.keep_spin)
        thr.addWidget(QtWidgets.QLabel(self.tr("复核上界")))
        self.review_spin = self._spin(0.0, 1.0, 0.05, 0.70)
        thr.addWidget(self.review_spin)
        self.rescan_btn = QtWidgets.QPushButton(self.tr("重新扫描"))
        thr.addWidget(self.rescan_btn)
        thr.addStretch(1)
        self.summary_label = QtWidgets.QLabel("")
        thr.addWidget(self.summary_label)
        root.addLayout(thr)

        # 快捷勾选
        quick = QtWidgets.QHBoxLayout()
        self.btn_all_keep = QtWidgets.QPushButton(self.tr("勾选所有 keep"))
        self.btn_add_review = QtWidgets.QPushButton(self.tr("加选所有 review"))
        self.btn_clear = QtWidgets.QPushButton(self.tr("清空勾选"))
        for b in (self.btn_all_keep, self.btn_add_review, self.btn_clear):
            quick.addWidget(b)
        quick.addStretch(1)
        root.addLayout(quick)

        # 分诊表
        self.table = QtWidgets.QTableWidget(0, len(self.COLS), self)
        self.table.setHorizontalHeaderLabels(
            [self.tr(c) for c in self.COLS]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for c in (0, 2, 3, 4, 5):
            hh.setSectionResizeMode(
                c, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )
        root.addWidget(self.table, 1)

        # 导出选项
        opt = QtWidgets.QHBoxLayout()
        opt.addWidget(QtWidgets.QLabel(self.tr("输出目录")))
        self.out_edit = QtWidgets.QLineEdit()
        opt.addWidget(self.out_edit, 1)
        self.browse_btn = QtWidgets.QPushButton(self.tr("浏览"))
        opt.addWidget(self.browse_btn)
        opt.addWidget(QtWidgets.QLabel(self.tr("val 比例")))
        self.val_spin = self._spin(0.0, 0.9, 0.05, 0.20)
        opt.addWidget(self.val_spin)
        self.grow_val_chk = QtWidgets.QCheckBox(self.tr("val 只增"))
        opt.addWidget(self.grow_val_chk)
        opt.addWidget(QtWidgets.QLabel(self.tr("每类入val")))
        self.per_class_val_spin = QtWidgets.QSpinBox()
        self.per_class_val_spin.setRange(0, 10)
        self.per_class_val_spin.setValue(1)
        self.per_class_val_spin.setToolTip(self.tr("保证每个类至少 N 个实例进验证集(小数据集建议 1);0=关闭"))
        opt.addWidget(self.per_class_val_spin)
        self.link_chk = QtWidgets.QCheckBox(self.tr("软链接"))
        opt.addWidget(self.link_chk)
        root.addLayout(opt)

        # 从 COCO128 注入数据(练手/补样本:按类名映射进 train/val)
        aug = QtWidgets.QHBoxLayout()
        aug.addWidget(QtWidgets.QLabel(self.tr("COCO目录")))
        self.coco_edit = QtWidgets.QLineEdit()
        self.coco_edit.setPlaceholderText(self.tr("如 D:\\code\\yolo\\datasets\\coco128"))
        aug.addWidget(self.coco_edit, 1)
        self.coco_browse_btn = QtWidgets.QPushButton(self.tr("浏览"))
        aug.addWidget(self.coco_browse_btn)
        aug.addWidget(QtWidgets.QLabel(self.tr("每类上限")))
        self.coco_maxpc_spin = QtWidgets.QSpinBox()
        self.coco_maxpc_spin.setRange(0, 1000)
        self.coco_maxpc_spin.setValue(0)
        self.coco_maxpc_spin.setToolTip(self.tr("每类最多注入多少张图(0=不限;想压 person 可设 15)"))
        aug.addWidget(self.coco_maxpc_spin)
        aug.addWidget(QtWidgets.QLabel(self.tr("每类入val")))
        self.coco_pcv_spin = QtWidgets.QSpinBox()
        self.coco_pcv_spin.setRange(0, 10)
        self.coco_pcv_spin.setValue(2)
        aug.addWidget(self.coco_pcv_spin)
        self.coco_preview_btn = QtWidgets.QPushButton(self.tr("预览注入"))
        self.coco_apply_btn = QtWidgets.QPushButton(self.tr("注入数据"))
        aug.addWidget(self.coco_preview_btn)
        aug.addWidget(self.coco_apply_btn)
        root.addLayout(aug)

        # 清理审查类(把 AI_REVIEW_* / missing_* 消化回真实类或删除,压缩类别表)
        clean = QtWidgets.QHBoxLayout()
        clean.addWidget(QtWidgets.QLabel(self.tr("清理审查类")))
        tip = QtWidgets.QLabel(self.tr("把 AI_REVIEW_* / missing_* 归并回真实类或删除(自动备份)"))
        tip.setStyleSheet("color:#8a93a3;")
        clean.addWidget(tip, 1)
        self.clean_preview_btn = QtWidgets.QPushButton(self.tr("预览清理"))
        self.clean_apply_btn = QtWidgets.QPushButton(self.tr("执行清理"))
        clean.addWidget(self.clean_preview_btn)
        clean.addWidget(self.clean_apply_btn)
        root.addLayout(clean)

        # 进度 + 日志
        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(True)
        root.addWidget(self.progress)
        self.log_list = QtWidgets.QListWidget()
        self.log_list.setMaximumHeight(120)
        self.log_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        root.addWidget(self.log_list)

        # 底部按钮
        bottom = QtWidgets.QHBoxLayout()
        bottom.addStretch(1)
        self.cancel_btn = QtWidgets.QPushButton(self.tr("关闭"))
        self.export_btn = QtWidgets.QPushButton(self.tr("导出发布"))
        try:
            self.cancel_btn.setStyleSheet(get_cancel_btn_style())
            self.export_btn.setStyleSheet(get_ok_btn_style())
        except Exception:
            pass
        self.export_btn.setDefault(True)
        bottom.addWidget(self.cancel_btn)
        bottom.addWidget(self.export_btn)
        root.addLayout(bottom)

        # 信号
        self.choose_src_btn.clicked.connect(self._choose_src)
        self.rescan_btn.clicked.connect(self._rescan)
        self.btn_all_keep.clicked.connect(lambda: self._bulk_check("keep"))
        self.btn_add_review.clicked.connect(
            lambda: self._bulk_check("review", add=True)
        )
        self.btn_clear.clicked.connect(lambda: self._bulk_check(None))
        self.browse_btn.clicked.connect(self._choose_out)
        self.cancel_btn.clicked.connect(self.reject)
        self.export_btn.clicked.connect(self._on_export)
        self.coco_browse_btn.clicked.connect(self._choose_coco)
        self.coco_preview_btn.clicked.connect(lambda: self._on_augment(apply=False))
        self.coco_apply_btn.clicked.connect(lambda: self._on_augment(apply=True))
        self.clean_preview_btn.clicked.connect(lambda: self._on_clean(apply=False))
        self.clean_apply_btn.clicked.connect(lambda: self._on_clean(apply=True))

    @staticmethod
    def _spin(lo, hi, step, val):
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setValue(val)
        s.setDecimals(2)
        return s

    # -------------------------------------------------------------- 数据
    def _image_dir(self):
        p = self.parent_widget
        if getattr(p, "filename", None):
            return osp.dirname(p.filename)
        return getattr(p, "last_open_dir", None)

    def _settings(self):
        """SelfEvolvingRecognition 主窗口的 QSettings 对象(用于持久化输出目录)。"""
        return getattr(self.parent_widget, "settings", None)

    def _restore_output_dir(self):
        st = self._settings()
        if st is None:
            return
        saved = st.value(self.SETTINGS_OUTPUT_KEY, "")
        if saved:
            self.out_edit.setText(str(saved))

    def _save_output_dir(self, d):
        """把用户设的输出目录写进 QSettings;第 2 步训练据此自动定位数据集。"""
        st = self._settings()
        if st is not None and d:
            st.setValue(self.SETTINGS_OUTPUT_KEY, d)

    def _default_source_dir(self):
        """猜测标注所在目录:优先独立标签目录,其次当前图片目录。"""
        p = self.parent_widget
        out = getattr(p, "output_dir", None)
        if out:
            return out
        if getattr(p, "filename", None):
            return osp.dirname(p.filename)
        return getattr(p, "last_open_dir", None) or ""

    def _choose_src(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, self.tr("选择标注目录")
        )
        if d:
            self.src_edit.setText(d)
            self._rescan()

    def _collect_json_files(self):
        from glob import glob

        src = self.src_edit.text().strip()
        if not src:
            return []
        return sorted(glob(osp.join(src, "**", "*.json"), recursive=True))

    def _rescan(self):
        src = self.src_edit.text().strip()
        if not src:
            self._log(
                self.tr("请先在主界面『打开目录』并自动标注,或用上方『选择目录』指定标注所在文件夹"),
                error=True,
            )
            return
        json_files = self._collect_json_files()
        if not json_files:
            self._log(self.tr("该目录下没找到任何 .json 标注:") + src, error=True)
            self._log(
                self.tr("确认这里有自动标注生成的 .json(默认与图片同目录)"),
                error=True,
            )
            return
        cfg = QualityConfig(
            score_keep=self.keep_spin.value(),
            score_review_hi=self.review_spin.value(),
        )
        self.records = scan_paths(
            json_files, cfg, images_dir=self._image_dir()
        )
        self._fill_table()
        c = {"keep": 0, "review": 0, "reject": 0}
        for r in self.records:
            c[r.verdict] += 1
        self.summary_label.setText(
            self.tr("共 %d  keep %d  review %d  reject %d")
            % (len(self.records), c["keep"], c["review"], c["reject"])
        )

    def _fill_table(self):
        self.table.setRowCount(0)
        for rec in self.records:
            row = self.table.rowCount()
            self.table.insertRow(row)

            chk = QtWidgets.QTableWidgetItem()
            if rec.verdict == "reject":
                # 0 框不能发布:显示禁用的勾选框
                chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable)
                chk.setCheckState(Qt.CheckState.Unchecked)
            else:
                chk.setFlags(
                    Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
                )
                chk.setCheckState(
                    Qt.CheckState.Checked
                    if rec.verdict == "keep"
                    else Qt.CheckState.Unchecked
                )
            self.table.setItem(row, 0, chk)

            name = (
                osp.basename(rec.image_path)
                if rec.image_path
                else osp.basename(rec.json_path) + self.tr(" (缺图)")
            )
            min_s = "" if rec.min_score is None else f"{rec.min_score:.2f}"
            cells = [name, str(rec.n_boxes), str(rec.n_kept), min_s, rec.verdict]
            for j, text in enumerate(cells, start=1):
                it = QtWidgets.QTableWidgetItem(text)
                if j in (2, 3, 4, 5):
                    it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, j, it)

            color = _VERDICT_BG.get(rec.verdict)
            if color is not None:
                for j in range(len(self.COLS)):
                    self.table.item(row, j).setBackground(color)

    def _bulk_check(self, verdict, add=False):
        for row, rec in enumerate(self.records):
            if rec.verdict == "reject":
                continue
            item = self.table.item(row, 0)
            if verdict is None:
                item.setCheckState(Qt.CheckState.Unchecked)
            elif rec.verdict == verdict:
                item.setCheckState(Qt.CheckState.Checked)
            elif not add:
                item.setCheckState(Qt.CheckState.Unchecked)

    def _choose_out(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, self.tr("选择输出目录")
        )
        if d:
            self.out_edit.setText(d)
            self._save_output_dir(d)

    # ----------------------------------------------------------- 日志/进度
    def _log(self, msg, error=False):
        it = QtWidgets.QListWidgetItem(("✗ " if error else "• ") + msg)
        if error:
            it.setForeground(QtGui.QColor(200, 60, 60))
        self.log_list.addItem(it)
        self.log_list.scrollToBottom()

    # --------------------------------------------------------------- 导出
    def _choose_coco(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, self.tr("选择 COCO 格式数据集目录(如 coco128)"))
        if d:
            self.coco_edit.setText(d)

    def _augment_dataset_dir(self):
        """注入目标 = 数据集目录。优先用输出目录框,空则用已保存的发布目录。"""
        d = self.out_edit.text().strip()
        if d:
            return d
        st = getattr(self.parent_widget, "settings", None)
        if st is not None:
            v = st.value("publish/output_dir", "")
            if v:
                return str(v)
        return ""

    def _on_augment(self, apply=False):
        coco = self.coco_edit.text().strip()
        ds = self._augment_dataset_dir()
        if not coco or not osp.isdir(coco):
            self._log(self.tr("请先选择有效的 COCO 数据集目录"), error=True)
            return
        if not ds or not osp.exists(osp.join(ds, "classes.txt")):
            self._log(self.tr("数据集目录无效(需含 classes.txt);请先在上方设置/发布数据集"), error=True)
            return
        try:
            from anylabeling.services.auto_training import augment_from_coco as AUG
        except ImportError:
            import augment_from_coco as AUG

        argv = ["--dataset", ds, "--coco", coco,
                "--per-class-val", str(self.coco_pcv_spin.value())]
        if self.coco_maxpc_spin.value() > 0:
            argv += ["--max-per-class", str(self.coco_maxpc_spin.value())]
        if apply:
            argv += ["--apply"]

        # augment 的输出走 print;临时把 stdout 接到日志面板
        import io
        import contextlib

        self.coco_preview_btn.setEnabled(False)
        self.coco_apply_btn.setEnabled(False)
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(Qt.CursorShape.WaitCursor))
        buf = io.StringIO()
        rc = 1
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                rc = AUG.main(argv)
        except Exception as e:  # noqa: BLE001
            logger.error(f"augment failed: {e}")
            self._log(self.tr("注入出错: ") + str(e), error=True)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.coco_preview_btn.setEnabled(True)
            self.coco_apply_btn.setEnabled(True)

        for line in buf.getvalue().splitlines():
            if line.strip():
                self._log(line)
        if apply and rc == 0:
            self._log(self.tr("注入完成。回到 Model Registry 训练一轮即可使用新数据。"))
            QtWidgets.QMessageBox.information(
                self, self.tr("注入完成"),
                self.tr("已把 COCO 数据注入到:\n") + ds +
                self.tr("\n\n现在可以去 Model Registry 训练一轮。"))
        elif not apply and rc == 0:
            self._log(self.tr("以上为预览。确认无误后点『注入数据』真正写入。"))

    def _on_clean(self, apply=False):
        ds = self._augment_dataset_dir()
        if not ds or not osp.exists(osp.join(ds, "classes.txt")):
            self._log(self.tr("数据集目录无效(需含 classes.txt);请先在上方设置/发布数据集"), error=True)
            return
        try:
            from anylabeling.services.auto_training import clean_review_classes as CLEAN
        except ImportError:
            import clean_review_classes as CLEAN

        argv = ["--dataset", ds]
        if apply:
            argv += ["--apply"]

        import io
        import contextlib

        self.clean_preview_btn.setEnabled(False)
        self.clean_apply_btn.setEnabled(False)
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(Qt.CursorShape.WaitCursor))
        buf = io.StringIO()
        rc = 1
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                rc = CLEAN.main(argv)
        except Exception as e:  # noqa: BLE001
            logger.error(f"clean review classes failed: {e}")
            self._log(self.tr("清理出错: ") + str(e), error=True)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.clean_preview_btn.setEnabled(True)
            self.clean_apply_btn.setEnabled(True)

        for line in buf.getvalue().splitlines():
            if line.strip():
                self._log(line)
        if apply and rc == 0:
            self._log(self.tr("清理完成(已自动备份)。可继续注入数据 / 去训练。"))
            QtWidgets.QMessageBox.information(
                self, self.tr("清理完成"),
                self.tr("已清理审查类并压缩类别表。\n原始 labels/classes 已备份到数据集下 _backup_clean_* 目录。"))
        elif not apply and rc == 0:
            self._log(self.tr("以上为预览。确认无误后点『执行清理』(会先自动备份)。"))

    def _on_export(self):
        out = self.out_edit.text().strip()
        if not out:
            self._log(self.tr("请先选择输出目录"), error=True)
            return
        chosen = [
            rec
            for row, rec in enumerate(self.records)
            if self.table.item(row, 0).checkState() == Qt.CheckState.Checked
        ]
        if not chosen:
            self._log(self.tr("没有勾选任何要发布的图"), error=True)
            return

        self.export_btn.setEnabled(False)
        self.rescan_btn.setEnabled(False)
        self.progress.setRange(0, len(chosen))
        self.progress.setValue(0)
        QtWidgets.QApplication.setOverrideCursor(
            QtGui.QCursor(Qt.CursorShape.WaitCursor)
        )

        def cb(done, total, stem):
            self.progress.setValue(done)
            QtWidgets.QApplication.processEvents()

        s = {"ok": False}
        try:
            s = export_records(
                chosen,
                out,
                val_ratio=self.val_spin.value(),
                grow_val=self.grow_val_chk.isChecked(),
                link=self.link_chk.isChecked(),
                verify_images=True,
                progress_cb=cb,
                per_class_val=self.per_class_val_spin.value(),
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"publish export failed: {e}")
            self._log(self.tr("导出出错: ") + str(e), error=True)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.export_btn.setEnabled(True)
            self.rescan_btn.setEnabled(True)

        if s.get("ok"):
            self._save_output_dir(s["out"])
            self._log(
                self.tr("导出完成: ")
                + f"train={s['n_train']} val={s['n_val']} "
                + self.tr("类别=")
                + str(s["n_classes"])
            )
            self._log(self.tr("数据集: ") + s["out"])
            self._log(
                self.tr("可直接训练: ")
                + "yolo detect train data=<dataset>/data.yaml model=yolo11n.pt"
            )
            QtWidgets.QMessageBox.information(
                self,
                self.tr("发布完成"),
                self.tr("已导出到:\n")
                + s["out"]
                + f"\n\ntrain={s['n_train']}  val={s['n_val']}  "
                + self.tr("类别=")
                + str(s["n_classes"]),
            )
        elif s.get("msg"):
            self._log(s["msg"], error=True)
