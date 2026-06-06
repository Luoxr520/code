# -*- coding: utf-8 -*-
"""
registry_dialog.py — 自我演化检测系统 · 第 2 步 GUI(模型版本面板)

放在 anylabeling/views/labeling/widgets/ 下,挂到「训练」菜单。
功能:
- 列出注册表里所有版本(mAP50-95 / mAP50 / 状态 / 时间 / 备注,★=当前线上);
- 训练一轮:子进程跑 train_with_registry train,训完自动评测+登记+过闸门(更好才上线);
- 登记最近一次训练:子进程跑 train_with_registry resume —— 训练已完成只补评测+登记(不重训),
  也是「上次训练中途崩了/eval 没跑」的一键恢复;
- 回退到选中版本:进程内直接改注册表 current(一键回退);
- 重的训练/评测走子进程(torch 不进 GUI 进程,和现有 TrainingManager 一样隔离),日志实时显示。

数据集目录默认读 SelfEvolvingRecognition 持久化的发布输出目录(QSettings publish/output_dir)。
"""
import os.path as osp
import re
import sys

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QProcess
from PyQt6.QtWidgets import QDialog

from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.style import (
    get_dialog_style,
    get_ok_btn_style,
    get_cancel_btn_style,
)

try:
    from anylabeling.services.auto_training.model_registry import ModelRegistry
    _REG_OK = True
    _REG_ERR = ""
except Exception as _e:  # noqa: BLE001
    _REG_OK = False
    _REG_ERR = str(_e)

_TRAIN_MODULE = "anylabeling.services.auto_training.train_with_registry"

_STATUS_BG = {
    "promoted": QtGui.QColor(60, 160, 90, 55),
    "superseded": QtGui.QColor(140, 140, 140, 40),
    "rejected": QtGui.QColor(200, 90, 60, 45),
    "seed": QtGui.QColor(80, 110, 200, 40),
}


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class _LogHighlighter(QtGui.QSyntaxHighlighter):
    """按行内容给日志上色,做出终端观感。"""

    def __init__(self, doc):
        super().__init__(doc)

        def fmt(color, bold=False):
            f = QtGui.QTextCharFormat()
            f.setForeground(QtGui.QColor(color))
            if bold:
                f.setFontWeight(QtGui.QFont.Weight.Bold)
            return f

        self.f_prompt = fmt("#4ec9b0", True)   # $ 命令
        self.f_ok = fmt("#73c990", True)       # 成功 / 已上线 / 完成
        self.f_err = fmt("#ff6b6b", True)      # 报错 / Traceback
        self.f_gate = fmt("#5aa9ff")           # [闸门]/[resume]/[train]...
        self.f_warn = fmt("#e6c07b")           # WARNING
        self.f_dim = fmt("#8a8f98")            # 进度 / 信息 / 分隔线

    def highlightBlock(self, text):
        t = text
        low = t.lower()
        if t.startswith("$ "):
            return self.setFormat(0, len(t), self.f_prompt)
        if t and all(c in "─ " for c in t):
            return self.setFormat(0, len(t), self.f_dim)
        if t.lstrip().startswith("✗") or "traceback" in low or "error" in low or "exception" in low:
            return self.setFormat(0, len(t), self.f_err)
        if "已上线" in t or "✅" in t or t.rstrip().endswith("完成"):
            return self.setFormat(0, len(t), self.f_ok)
        if any(t.startswith(p) for p in ("[闸门]", "[结果]", "[resume]", "[train]", "[eval]", "[serve]")):
            return self.setFormat(0, len(t), self.f_gate)
        if "warning" in low:
            return self.setFormat(0, len(t), self.f_warn)
        if "epoch" in low or "%" in t or "it/s" in low or "map" in low or "ultralytics" in low:
            return self.setFormat(0, len(t), self.f_dim)


def parse_epoch_line(line):
    """'@EPOCH <ep> <map50> <map5095>' -> (ep, map50, map5095) | None(纯函数,便于测试)。"""
    s = line.strip()
    if not s.startswith("@EPOCH"):
        return None
    parts = s.split()
    if len(parts) < 4:
        return None
    try:
        ep = int(parts[1])
    except ValueError:
        return None

    def num(x):
        try:
            v = float(x)
        except ValueError:
            return None
        if v != v or v in (float("inf"), float("-inf")):  # NaN / Inf -> 视为缺失
            return None
        return v

    return ep, num(parts[2]), num(parts[3])


class MetricChart(QtWidgets.QWidget):
    """每轮 mAP 实时折线图:多次训练同图叠加(不同颜色)、平滑曲线、最新点柔和滑入动画。
    纯 QPainter 实现,无需 QtCharts/matplotlib。"""

    PALETTE = ["#5aa9ff", "#73c990", "#e6c07b", "#ff8087",
               "#b48ead", "#56c2c2", "#d19a66", "#c678dd"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.runs = {}        # run_id -> [(epoch, val), ...]
        self.order = []       # run_id 顺序(决定配色)
        self._active = None   # 正在增长的 run
        self._t = 1.0         # 最新点滑入动画进度 [0,1]
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self.setMinimumHeight(240)

    # ---- 数据接口 ----
    def clear_all(self):
        self.runs.clear()
        self.order.clear()
        self._active = None
        self._t = 1.0
        self._timer.stop()
        self.update()

    def start_run(self, run_id):
        if run_id not in self.runs:
            self.runs[run_id] = []
            self.order.append(run_id)
        self._active = run_id
        self.update()

    def add_point(self, run_id, epoch, val):
        if val is None or val != val:   # None 或 NaN 直接跳过
            return
        if run_id not in self.runs:
            self.start_run(run_id)
        self.runs[run_id].append((int(epoch), float(val)))
        self._active = run_id
        self._t = 0.0
        if not self._timer.isActive():
            self._timer.start()

    def _tick(self):
        self._t = min(1.0, self._t + 0.07)
        if self._t >= 1.0:
            self._timer.stop()
        self.update()

    @staticmethod
    def _ease(t):
        return 1 - (1 - t) ** 3  # easeOutCubic

    def _color(self, run_id):
        i = self.order.index(run_id) if run_id in self.order else 0
        return QtGui.QColor(self.PALETTE[i % len(self.PALETTE)])

    def _bounds(self):
        max_ep, max_v = 1, 0.0
        for pts in self.runs.values():
            for (e, v) in pts:
                max_ep = max(max_ep, e)
                max_v = max(max_v, v)
        return max_ep, max(max_v * 1.15, 0.1)

    @staticmethod
    def _spline(px):
        """Catmull-Rom -> 三次贝塞尔平滑路径。控制点 y 钳制在相邻数据点值域内,
        避免在"长期平→突然阶跃"拐点处过冲(钻到 0 轴以下或冲过峰顶)。"""
        path = QtGui.QPainterPath()
        n = len(px)
        if n == 0:
            return path
        path.moveTo(px[0][0], px[0][1])
        if n == 1:
            return path
        if n == 2:
            path.lineTo(px[1][0], px[1][1])
            return path
        for i in range(n - 1):
            p0 = px[i - 1] if i > 0 else px[0]
            p1, p2 = px[i], px[i + 1]
            p3 = px[i + 2] if i + 2 < n else px[-1]
            c1x = p1[0] + (p2[0] - p0[0]) / 6.0
            c1y = p1[1] + (p2[1] - p0[1]) / 6.0
            c2x = p2[0] - (p3[0] - p1[0]) / 6.0
            c2y = p2[1] - (p3[1] - p1[1]) / 6.0
            # 单调钳制:把控制点 y 限制在本段两端点之间,杜绝过冲
            lo, hi = min(p1[1], p2[1]), max(p1[1], p2[1])
            c1y = max(lo, min(hi, c1y))
            c2y = max(lo, min(hi, c2y))
            path.cubicTo(c1x, c1y, c2x, c2y, p2[0], p2[1])
        return path

    # ---- 绘制 ----
    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        W, H = self.width(), self.height()
        ml, mr, mt, mb = 44, 14, 16, 24
        x0, y0, x1, y1 = ml, mt, W - mr, H - mb
        pw, ph = max(1, x1 - x0), max(1, y1 - y0)

        p.fillRect(self.rect(), QtGui.QColor("#0b0d11"))
        p.setPen(QtGui.QPen(QtGui.QColor("#222834"), 1))
        p.drawRoundedRect(0, 0, W - 1, H - 1, 8, 8)

        if not any(self.runs.values()):
            p.setPen(QtGui.QColor("#5a6473"))
            p.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter,
                       self.tr("训练开始后这里实时显示每轮 mAP"))
            p.end()
            return

        max_ep, max_v = self._bounds()

        def X(e):
            return x0 + (e - 1) / max(1, (max_ep - 1)) * pw if max_ep > 1 else x0 + pw * 0.5

        def Y(v):
            return y1 - (v / max_v) * ph

        font = p.font()
        font.setPointSize(8)
        p.setFont(font)
        grid = 4
        for i in range(grid + 1):
            gy = y0 + ph * i / grid
            p.setPen(QtGui.QPen(QtGui.QColor("#171c24"), 1))
            p.drawLine(int(x0), int(gy), int(x1), int(gy))
            p.setPen(QtGui.QColor("#4a5360"))
            p.drawText(2, int(gy) + 4, f"{max_v * (1 - i / grid):.2f}")
        p.setPen(QtGui.QColor("#4a5360"))
        p.drawText(int(x0), int(y1) + 16, "1")
        p.drawText(int(x1) - 20, int(y1) + 16, str(max_ep))

        for rid in self.order:
            pts = self.runs.get(rid, [])
            if not pts:
                continue
            col = self._color(rid)
            px = []
            for idx, (e, v) in enumerate(pts):
                xe, yv = X(e), Y(v)
                if rid == self._active and idx == len(pts) - 1 and len(pts) >= 2 and self._t < 1.0:
                    pe, pv = pts[idx - 1]
                    te = self._ease(self._t)
                    xe = X(pe) + (X(e) - X(pe)) * te
                    yv = Y(pv) + (Y(v) - Y(pv)) * te
                px.append((xe, yv))
            pen = QtGui.QPen(col, 2.0)
            pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.drawPath(self._spline(px))
            lx, ly = px[-1]
            glow = QtGui.QColor(col)
            glow.setAlpha(60)
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(glow)
            p.drawEllipse(QtCore.QPointF(lx, ly), 6, 6)
            p.setBrush(col)
            p.drawEllipse(QtCore.QPointF(lx, ly), 3.2, 3.2)

        # 图例(右上),显示各次最新值
        fm = QtGui.QFontMetrics(font)
        ly = y0 + 4
        for rid in self.order:
            pts = self.runs.get(rid, [])
            last = pts[-1][1] if pts else 0.0
            label = f"{rid}  {last:.3f}"
            tw = fm.horizontalAdvance(label)
            col = self._color(rid)
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(col)
            p.drawRoundedRect(int(x1 - tw - 18), int(ly), 9, 9, 2, 2)
            p.setPen(QtGui.QColor("#aeb6c2"))
            p.drawText(int(x1 - tw - 5), int(ly) + 9, label)
            ly += 15
        p.end()


class ModelRegistryDialog(QDialog):
    COLS = ["", "版本", "mAP50-95", "mAP50", "状态", "来源", "时间", "备注"]

    def __init__(self, parent):
        super().__init__(parent)
        self.parent_widget = parent
        self.proc = None
        self._tail = ""
        self._cur_run = None      # 当前训练对应的折线 id
        self._run_seq = 0         # 训练次数计数(每次一种颜色)
        self._setup_ui()
        if not _REG_OK:
            self._log(self.tr("无法导入 model_registry: ") + _REG_ERR, error=True)
            self._set_busy(True)  # 禁用操作按钮
            return
        self.ds_edit.setText(self._default_dataset())
        # 便利:若数据集里有 coco128,预填到回放框(默认不勾选)
        ds0 = self.ds_edit.text().strip()
        if ds0 and osp.isdir(osp.join(ds0, "coco128")) and not self.replay_edit.text().strip():
            self.replay_edit.setText(osp.join(ds0, "coco128"))
        self.refresh()

    # ------------------------------------------------------------------ UI
    def _setup_ui(self):
        self.setWindowTitle(self.tr("Model Registry"))
        self.resize(1280, 880)
        self.showMaximized()
        try:
            self.setStyleSheet(get_dialog_style())
        except Exception:
            pass

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        # 数据集目录
        ds_row = QtWidgets.QHBoxLayout()
        ds_row.addWidget(QtWidgets.QLabel(self.tr("数据集目录")))
        self.ds_edit = QtWidgets.QLineEdit()
        self.ds_edit.setPlaceholderText(self.tr("发布面板导出的数据集目录(含 data.yaml / registry)"))
        ds_row.addWidget(self.ds_edit, 1)
        self.choose_ds_btn = QtWidgets.QPushButton(self.tr("选择"))
        self.refresh_btn = QtWidgets.QPushButton(self.tr("刷新"))
        ds_row.addWidget(self.choose_ds_btn)
        ds_row.addWidget(self.refresh_btn)
        root.addLayout(ds_row)

        self.summary_label = QtWidgets.QLabel("")
        root.addWidget(self.summary_label)

        # 版本表
        self.table = QtWidgets.QTableWidget(0, len(self.COLS), self)
        self.table.setHorizontalHeaderLabels([self.tr(c) for c in self.COLS])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(7, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for c in (0, 1, 2, 3, 4, 5, 6):
            hh.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        root.addWidget(self.table, 2)

        # 训练参数
        opt = QtWidgets.QHBoxLayout()
        opt.addWidget(QtWidgets.QLabel(self.tr("起始权重")))
        self.base_edit = QtWidgets.QLineEdit("yolo11n.pt")
        self.base_edit.setMaximumWidth(180)
        opt.addWidget(self.base_edit)
        opt.addWidget(QtWidgets.QLabel(self.tr("epochs")))
        self.epochs_spin = QtWidgets.QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        opt.addWidget(self.epochs_spin)
        opt.addWidget(QtWidgets.QLabel(self.tr("device")))
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItems(["0", "cpu", self.tr("自动")])
        opt.addWidget(self.device_combo)
        opt.addWidget(QtWidgets.QLabel(self.tr("workers")))
        self.workers_spin = QtWidgets.QSpinBox()
        self.workers_spin.setRange(0, 32)
        self.workers_spin.setValue(0)
        opt.addWidget(self.workers_spin)
        opt.addStretch(1)
        root.addLayout(opt)

        # 抗遗忘选项
        af = QtWidgets.QHBoxLayout()
        self.replay_check = QtWidgets.QCheckBox(self.tr("回放混入"))
        self.replay_check.setToolTip(self.tr("把这些目录的样本(按类名重映射)混进训练,抗遗忘"))
        af.addWidget(self.replay_check)
        self.replay_edit = QtWidgets.QLineEdit()
        self.replay_edit.setPlaceholderText(self.tr("回放源目录,多个用 ; 分隔(如 coco128;distill)"))
        af.addWidget(self.replay_edit, 1)
        self.replay_add_btn = QtWidgets.QPushButton(self.tr("加目录"))
        self.distill_btn = QtWidgets.QPushButton(self.tr("生成蒸馏(LwF)"))
        self.distill_btn.setToolTip(self.tr("当前模型给所选(无标注)图片打伪标注 -> <dataset>/distill,并加入回放"))
        af.addWidget(self.replay_add_btn)
        af.addWidget(self.distill_btn)
        self.freeze_check = QtWidgets.QCheckBox(self.tr("冻结主干"))
        self.freeze_check.setToolTip(self.tr("冻结前 N 层,保留预训练特征"))
        af.addWidget(self.freeze_check)
        self.freeze_spin = QtWidgets.QSpinBox()
        self.freeze_spin.setRange(1, 23)
        self.freeze_spin.setValue(10)
        af.addWidget(self.freeze_spin)
        af.addWidget(QtWidgets.QLabel("lr0"))
        self.lr0_edit = QtWidgets.QLineEdit()
        self.lr0_edit.setMaximumWidth(80)
        self.lr0_edit.setPlaceholderText(self.tr("可空"))
        af.addWidget(self.lr0_edit)
        af.addWidget(QtWidgets.QLabel(self.tr("优化器")))
        self.optimizer_combo = QtWidgets.QComboBox()
        self.optimizer_combo.addItems(["auto", "SGD", "AdamW"])
        self.optimizer_combo.setToolTip(self.tr("auto 会忽略 lr0;想让 lr0 生效就选 SGD 或 AdamW"))
        af.addWidget(self.optimizer_combo)
        root.addLayout(af)

        # 操作按钮
        act = QtWidgets.QHBoxLayout()
        self.train_btn = QtWidgets.QPushButton(self.tr("训练一轮"))
        self.resume_btn = QtWidgets.QPushButton(self.tr("登记最近一次训练"))
        self.continue_btn = QtWidgets.QPushButton(self.tr("继续中断的训练"))
        self.rollback_btn = QtWidgets.QPushButton(self.tr("回退到选中版本"))
        self.reeval_btn = QtWidgets.QPushButton(self.tr("全部重评"))
        self.reeval_btn.setToolTip(self.tr("在当前验证集上重评所有版本并重判 current(换了验证集/注入数据后用)"))
        self.stop_btn = QtWidgets.QPushButton(self.tr("停止"))
        self.stop_btn.setEnabled(False)
        self.train_btn.setToolTip(self.tr("从当前线上模型继续训练一轮,训完自动评测+登记(更好才上线)"))
        self.resume_btn.setToolTip(self.tr("训练已完成、只剩评测时用:只补评测+登记,不重训"))
        self.continue_btn.setToolTip(self.tr("训练中途被打断时用:从 last.pt 续训到设定 epoch,再评测+登记"))
        self.rollback_btn.setToolTip(self.tr("把选中版本设为当前线上模型"))
        for b in (self.train_btn, self.resume_btn, self.continue_btn,
                  self.rollback_btn, self.reeval_btn, self.stop_btn):
            act.addWidget(b)
        act.addStretch(1)
        root.addLayout(act)

        # 日志(终端样式)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(8000)
        self.log.setMinimumHeight(180)
        self.log.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        log_font = QtGui.QFont("Consolas")
        log_font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        log_font.setPointSize(9)
        self.log.setFont(log_font)
        self.log.setStyleSheet(
            "QPlainTextEdit{background:#0f1115;color:#cdd3de;"
            "border:1px solid #2a2f3a;border-radius:8px;padding:8px;"
            "selection-background-color:#2d4f7c;}"
        )
        self._hl = _LogHighlighter(self.log.document())

        # 折线图(每轮 mAP)与日志并排显示
        self.chart = MetricChart()
        lower = QtWidgets.QHBoxLayout()
        lower.setSpacing(10)
        lower.addWidget(self.chart, 1)
        lower.addWidget(self.log, 1)
        root.addLayout(lower, 3)

        # 底部
        bottom = QtWidgets.QHBoxLayout()
        bottom.addStretch(1)
        self.close_btn = QtWidgets.QPushButton(self.tr("关闭"))
        try:
            self.close_btn.setStyleSheet(get_cancel_btn_style())
            self.train_btn.setStyleSheet(get_ok_btn_style())
        except Exception:
            pass
        bottom.addWidget(self.close_btn)
        root.addLayout(bottom)

        # 信号
        self.choose_ds_btn.clicked.connect(self._choose_ds)
        self.refresh_btn.clicked.connect(self.refresh)
        self.train_btn.clicked.connect(self._on_train)
        self.resume_btn.clicked.connect(self._on_resume)
        self.continue_btn.clicked.connect(self._on_continue)
        self.rollback_btn.clicked.connect(self._on_rollback)
        self.reeval_btn.clicked.connect(self._on_reeval)
        self.stop_btn.clicked.connect(self._on_stop)
        self.close_btn.clicked.connect(self.reject)
        self.table.itemChanged.connect(self._on_note_edited)
        self.replay_add_btn.clicked.connect(self._on_replay_add)
        self.distill_btn.clicked.connect(self._on_distill)

    # -------------------------------------------------------------- 数据
    def _default_dataset(self):
        st = getattr(self.parent_widget, "settings", None)
        if st is not None:
            v = st.value("publish/output_dir", "")
            if v:
                return str(v)
        return ""

    def _choose_ds(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, self.tr("选择数据集目录"))
        if d:
            self.ds_edit.setText(d)
            self.refresh()

    def refresh(self):
        ds = self.ds_edit.text().strip()
        self.table.blockSignals(True)   # 填充期间别触发 itemChanged(否则会误存备注)
        try:
            self.table.setRowCount(0)
            if not ds or not osp.isdir(ds):
                self.summary_label.setText(self.tr("请先选择数据集目录"))
                return
            try:
                reg = ModelRegistry(ds)
                rows = reg.list()
                metric = reg.metric
            except Exception as e:  # noqa: BLE001
                self._log(self.tr("读取注册表失败: ") + str(e), error=True)
                return
            cur = None
            for e in rows:
                self._append_row(e)
                if e.get("is_current"):
                    cur = e
            if not rows:
                self.summary_label.setText(self.tr("注册表为空。点『训练一轮』开始,或先用发布面板导出数据集。"))
            elif cur:
                self.summary_label.setText(
                    self.tr("当前线上: ") + f"{cur['id']}  {metric}={cur.get('metrics', {}).get(metric)}"
                )
        finally:
            self.table.blockSignals(False)

    def _append_row(self, e):
        row = self.table.rowCount()
        self.table.insertRow(row)
        is_cur = bool(e.get("is_current"))
        status = e.get("status", "")
        metrics = e.get("metrics", {}) or {}

        def cell(text, align_center=False, editable=False):
            it = QtWidgets.QTableWidgetItem("" if text is None else str(text))
            if align_center:
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            flags = it.flags()
            if editable:
                flags |= Qt.ItemFlag.ItemIsEditable
            else:
                flags &= ~Qt.ItemFlag.ItemIsEditable
            it.setFlags(flags)
            return it

        def fmt(x):
            return f"{x:.4f}" if isinstance(x, (int, float)) else "-"

        self.table.setItem(row, 0, cell("★" if is_cur else "", True))
        self.table.setItem(row, 1, cell(e.get("id"), True))
        self.table.setItem(row, 2, cell(fmt(metrics.get("map5095")), True))
        self.table.setItem(row, 3, cell(fmt(metrics.get("map50")), True))
        self.table.setItem(row, 4, cell(status, True))
        self.table.setItem(row, 5, cell(e.get("parent_id"), True))
        self.table.setItem(row, 6, cell(e.get("ts"), True))
        note_item = cell(e.get("note", ""), editable=True)   # 仅备注列可双击编辑
        note_item.setData(Qt.ItemDataRole.UserRole, e.get("id"))
        note_item.setToolTip(self.tr("双击编辑备注,回车保存"))
        self.table.setItem(row, 7, note_item)

        bg = _STATUS_BG["promoted"] if is_cur else _STATUS_BG.get(status)
        if bg is not None:
            for c in range(len(self.COLS)):
                self.table.item(row, c).setBackground(bg)

    def _selected_id(self):
        items = self.table.selectedItems()
        if not items:
            return None
        return self.table.item(items[0].row(), 1).text()

    # ----------------------------------------------------------- 子进程
    def _device_arg(self):
        d = self.device_combo.currentText()
        return None if d == self.tr("自动") else d

    def _run(self, sub_args, what):
        ds = self.ds_edit.text().strip()
        if not ds or not osp.isdir(ds):
            self._log(self.tr("请先选择有效的数据集目录"), error=True)
            return
        if self.proc is not None and self.proc.state() != QProcess.ProcessState.NotRunning:
            self._log(self.tr("已有任务在运行,请先停止"), error=True)
            return
        args = ["-u", "-m", _TRAIN_MODULE, "--dataset", ds] + sub_args
        self._tail = ""
        # 命令跳转:分隔线 + 提示符(直接写入,让高亮器把 $ 行染成提示符色)
        self.log.appendPlainText("─" * 64)
        self.log.appendPlainText(f"$ python -m {_TRAIN_MODULE} --dataset {ds} " + " ".join(sub_args))
        self.log.appendPlainText("")  # 给子进程输出留一行,避免回车覆盖掉命令行
        self.proc = QProcess(self)
        # 强制子进程用 UTF-8 输出,根治 Windows 下中文乱码
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        self.proc.setProcessEnvironment(env)
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._on_proc_output)
        self.proc.finished.connect(lambda code, st: self._on_proc_finished(code, what))
        self._set_busy(True)
        self.proc.start(sys.executable, args)

    def _antiforget_args(self):
        """把抗遗忘选项拼成 CLI 片段。"""
        import re

        extra = []
        if self.replay_check.isChecked():
            for d in re.split(r"[;\n]", self.replay_edit.text()):
                d = d.strip()
                if d:
                    extra += ["--replay", d]
        if self.freeze_check.isChecked():
            extra += ["--freeze", str(self.freeze_spin.value())]
        lr0 = self.lr0_edit.text().strip()
        if lr0:
            extra += ["--lr0", lr0]
        opt = self.optimizer_combo.currentText()
        if opt and opt != "auto":   # auto 是 ultralytics 默认,不必显式传
            extra += ["--optimizer", opt]
        return extra

    def _on_train(self):
        sub = ["train", "--epochs", str(self.epochs_spin.value()),
               "--workers", str(self.workers_spin.value()),
               "--base", self.base_edit.text().strip() or "yolo11n.pt"]
        dev = self._device_arg()
        if dev:
            sub += ["--device", dev]
        sub += self._antiforget_args()
        self._run_seq += 1
        self._cur_run = f"#{self._run_seq}"
        self.chart.start_run(self._cur_run)
        self._run(sub, self.tr("训练一轮"))

    def _on_replay_add(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, self.tr("选择回放源目录"))
        if not d:
            return
        cur = self.replay_edit.text().strip()
        self.replay_edit.setText((cur + ";" + d) if cur else d)
        self.replay_check.setChecked(True)

    def _on_distill(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, self.tr("选择要打伪标注的图片文件夹(无标注即可)"))
        if not d:
            return
        sub = ["distill", "--teacher", "current", "--images", d]
        dev = self._device_arg()
        if dev:
            sub += ["--device", dev]
        self._run(sub, self.tr("生成蒸馏伪标注(LwF)"))

    def _on_resume(self):
        sub = ["resume", "--workers", str(self.workers_spin.value())]
        dev = self._device_arg()
        if dev:
            sub += ["--device", dev]
        self._run(sub, self.tr("登记最近一次训练"))

    def _on_continue(self):
        sub = ["resume", "--continue-training", "--workers", str(self.workers_spin.value())]
        dev = self._device_arg()
        if dev:
            sub += ["--device", dev]
        # 续训 = 同一个模型从 last.pt 接着练,折线接在原来那条线上(不新建颜色)。
        # ultralytics resume 的 epoch 编号接续原 run,@EPOCH 会继续往后发,自然延伸曲线。
        if not self._cur_run:
            self._run_seq += 1
            self._cur_run = f"#{self._run_seq}"
        self.chart.start_run(self._cur_run)   # 已存在则只设为活动,不清空已有点
        self._log(self.tr("续训:在折线 %s 上继续(同一模型)") % self._cur_run)
        self._run(sub, self.tr("继续中断的训练"))

    def _on_stop(self):
        if self.proc is not None and self.proc.state() != QProcess.ProcessState.NotRunning:
            self.proc.kill()
            self._log(self.tr("已请求停止当前任务"))

    def _on_proc_output(self):
        if self.proc is None:
            return
        text = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text:
            self._emit(text)

    def _consume_line(self, line):
        """@EPOCH 行喂给折线图、不进日志;返回是否已消费。"""
        if not line.lstrip().startswith("@EPOCH"):
            return False
        r = parse_epoch_line(line)
        if r and self._cur_run:
            ep, map50, map5095 = r
            v = map5095 if map5095 is not None else map50  # 主画 mAP50-95
            self.chart.add_point(self._cur_run, ep, v)
        return True

    def _emit(self, raw):
        """稳妥写日志:去 ANSI,按行缓冲。每遇到 \\n 就把该行最终样子(\\r 之后的部分)整行追加。
        只追加、不删除,因此不会吃掉已写内容;进度条按行完成时整行显示。"""
        raw = _ANSI_RE.sub("", raw).replace("\x00", "")
        if not raw:
            return
        # 关键:Windows 子进程多用 \r\n 换行。先把 CRLF 归一成 \n,
        # 否则行尾那个 \r 会被当成进度回车,把整行内容吃成空行(这就是"不打印"的真因)。
        self._tail = (self._tail + raw).replace("\r\n", "\n")
        while "\n" in self._tail:
            line, self._tail = self._tail.split("\n", 1)
            if "\r" in line:           # 行内单独的回车 = 进度反复刷新,只保留最后一帧
                line = line.split("\r")[-1]
            if self._consume_line(line):
                continue
            self.log.appendPlainText(line)
        self.log.ensureCursorVisible()

    def _flush_tail(self):
        """进程结束时,把最后一段没有换行的残留也打出来。"""
        if not self._tail:
            return
        t = self._tail.replace("\r\n", "\n")
        self._tail = ""
        for line in t.split("\n"):
            if "\r" in line:
                line = line.split("\r")[-1]
            if self._consume_line(line):
                continue
            self.log.appendPlainText(line)

    def _on_proc_finished(self, code, what):
        self._flush_tail()
        self._set_busy(False)
        self._log(f"[{what}] " + (self.tr("完成") if code == 0 else self.tr("结束(退出码 %d)") % code))
        # 蒸馏成功后,把 <dataset>/distill 自动加进回放
        if code == 0 and "蒸馏" in what:
            ds = self.ds_edit.text().strip()
            dd = osp.join(ds, "distill")
            cur = self.replay_edit.text().strip()
            if dd not in cur:
                self.replay_edit.setText((cur + ";" + dd) if cur else dd)
            self.replay_check.setChecked(True)
            self._log(self.tr("已把蒸馏目录加入回放: ") + dd)
        self.refresh()

    # ----------------------------------------------------------- 备注编辑
    def _on_note_edited(self, item):
        if item.column() != 7:
            return
        ds = self.ds_edit.text().strip()
        rid = item.data(Qt.ItemDataRole.UserRole)
        if not rid or not ds:
            return
        try:
            ModelRegistry(ds).set_note(rid, item.text())
        except Exception as e:  # noqa: BLE001
            self._log(self.tr("保存备注失败: ") + str(e), error=True)
            return
        self._log(self.tr("已更新 %s 的备注") % rid)

    # ----------------------------------------------------------- 回退
    def _on_reeval(self):
        dev = self._device_arg()
        sub = ["reeval-all", "--workers", str(self.workers_spin.value())]
        if dev:
            sub += ["--device", dev]
        self._log(self.tr("在当前验证集上重评所有版本(换验证集后让分数可比)…"))
        self._run(sub, self.tr("全部重评"))

    def _on_rollback(self):
        ds = self.ds_edit.text().strip()
        rid = self._selected_id()
        if not rid:
            self._log(self.tr("请先在表里选中一个版本"), error=True)
            return
        try:
            reg = ModelRegistry(ds)
            reg.rollback(rid)
        except Exception as e:  # noqa: BLE001
            self._log(self.tr("回退失败: ") + str(e), error=True)
            return
        self._log(self.tr("已回退,当前线上 = ") + rid)
        self.refresh()

    # ----------------------------------------------------------- 杂项
    def _set_busy(self, busy):
        for b in (self.train_btn, self.resume_btn, self.continue_btn,
                  self.rollback_btn, self.reeval_btn, self.refresh_btn,
                  self.replay_add_btn, self.distill_btn):
            b.setEnabled(not busy and _REG_OK)
        self.stop_btn.setEnabled(busy)

    def _log(self, msg, error=False):
        prefix = "✗ " if error else "• "
        self.log.appendPlainText(prefix + msg)

    def reject(self):
        if self.proc is not None and self.proc.state() != QProcess.ProcessState.NotRunning:
            self.proc.kill()
        super().reject()

    def closeEvent(self, ev):
        # 非模态 + WA_DeleteOnClose 时,点窗口X走这里,确保正在跑的训练/重评进程被结束
        if self.proc is not None and self.proc.state() != QProcess.ProcessState.NotRunning:
            self.proc.kill()
        super().closeEvent(ev)
