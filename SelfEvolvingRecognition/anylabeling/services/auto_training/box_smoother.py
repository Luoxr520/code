# -*- coding: utf-8 -*-
"""
box_smoother.py — 自我演化检测系统 · 第 3b 步
检测框连续过渡:跨帧跟踪关联 + EMA 平滑 + 渐隐渐显。

输入:每帧的原始检测列表(model_server.infer 返回的 Det,或任何带
      .x1 .y1 .x2 .y2 .conf .cls .label 属性的对象)。
输出:每帧的"显示框"列表 SBox(带稳定 id、平滑后的 xyxy/conf、透明度 alpha)。

为什么能让画面平滑、换模型也不跳:
- 逐类贪心 IoU 关联,给同一目标稳定的 track id(跨帧不闪、不跳号);
- 在 (中心x,中心y,宽,高) 空间做 EMA,框平滑移动/收紧,而非逐帧瞬移;
- 透明度由 track 生命周期驱动:新目标渐入、消失目标渐出再移除;
- 热替换时新模型给的框靠 IoU 关联到原 track,EMA 把框从旧位置平滑滑到新位置
  —— 这正是"框越收越准"的过渡动画,且 track id 不变。

无运动外推:目标短暂丢失时框保持原位(由 alpha 渐隐),实现简单且够用。
关联这步是独立的,以后可替换成 ByteTrack(只改 _associate)。
"""
from __future__ import annotations

from collections import defaultdict, namedtuple

# 一个显示框:稳定 id + 平滑后的像素坐标 + 平滑置信度 + 透明度 + 类别
SBox = namedtuple("SBox", "id x1 y1 x2 y2 conf alpha cls label")


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _to_cxcywh(x1, y1, x2, y2):
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0,
            max(1e-3, x2 - x1), max(1e-3, y2 - y1))


def _to_xyxy(cx, cy, w, h):
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


class _Track:
    __slots__ = ("id", "cx", "cy", "w", "h", "conf", "cls", "label",
                 "hits", "tsu", "age")

    def __init__(self, tid, det):
        self.id = tid
        self.cx, self.cy, self.w, self.h = _to_cxcywh(det.x1, det.y1, det.x2, det.y2)
        self.conf = float(det.conf)
        self.cls = int(det.cls)
        self.label = det.label
        self.hits = 1
        self.tsu = 0        # time since update(连续未匹配的帧数)
        self.age = 1

    def xyxy(self):
        return _to_xyxy(self.cx, self.cy, self.w, self.h)

    def update(self, det, smooth):
        tcx, tcy, tw, th = _to_cxcywh(det.x1, det.y1, det.x2, det.y2)
        s = smooth
        self.cx += s * (tcx - self.cx)
        self.cy += s * (tcy - self.cy)
        self.w += s * (tw - self.w)
        self.h += s * (th - self.h)
        self.conf += s * (float(det.conf) - self.conf)
        self.label = det.label
        self.hits += 1
        self.tsu = 0
        self.age += 1


class BoxSmoother:
    def __init__(self, iou_match=0.3, smooth=0.4, fade_in=3, fade_out=6,
                 max_age=8, min_alpha=0.03):
        self.iou_match = iou_match
        self.smooth = smooth          # EMA 系数:大=跟手/抖,小=平滑/滞后
        self.fade_in = fade_in        # 渐入帧数
        self.fade_out = fade_out      # 渐出帧数
        self.max_age = max_age        # 连续丢失多少帧后移除
        self.min_alpha = min_alpha
        self.tracks = {}              # id -> _Track
        self._next_id = 1
        self.frame = 0

    def reset(self):
        self.tracks.clear()
        self._next_id = 1
        self.frame = 0

    def _associate(self, dets):
        """逐类贪心 IoU 匹配。返回 (matched: {track_id:det_idx}, unmatched_det_idx:set)。"""
        det_by_cls = defaultdict(list)
        for i, d in enumerate(dets):
            det_by_cls[int(d.cls)].append(i)
        trk_by_cls = defaultdict(list)
        for tid, t in self.tracks.items():
            trk_by_cls[t.cls].append(tid)

        matched = {}
        used_det = set()
        for cls, dis in det_by_cls.items():
            tis = trk_by_cls.get(cls, [])
            pairs = []
            for tid in tis:
                tb = self.tracks[tid].xyxy()
                for di in dis:
                    d = dets[di]
                    iou = _iou(tb, (d.x1, d.y1, d.x2, d.y2))
                    if iou >= self.iou_match:
                        pairs.append((iou, tid, di))
            pairs.sort(reverse=True)   # 先配 IoU 最高的
            used_trk = set()
            for iou, tid, di in pairs:
                if tid in used_trk or di in used_det:
                    continue
                used_trk.add(tid)
                used_det.add(di)
                matched[tid] = di
        unmatched_det = {i for i in range(len(dets)) if i not in used_det}
        return matched, unmatched_det

    def update(self, dets):
        """喂入一帧原始检测,返回该帧的显示框列表 [SBox,...]。"""
        self.frame += 1
        dets = list(dets)
        prev_ids = list(self.tracks.keys())

        matched, unmatched_det = self._associate(dets)

        for tid, di in matched.items():
            self.tracks[tid].update(dets[di], self.smooth)
        for i in unmatched_det:
            self.tracks[self._next_id] = _Track(self._next_id, dets[i])
            self._next_id += 1
        for tid in prev_ids:
            if tid not in matched:
                self.tracks[tid].tsu += 1
                self.tracks[tid].age += 1
        for tid in list(self.tracks.keys()):
            if self.tracks[tid].tsu > self.max_age:
                del self.tracks[tid]

        out = []
        for tid, t in self.tracks.items():
            a_in = min(1.0, t.hits / float(self.fade_in))
            a_out = max(0.0, 1.0 - t.tsu / float(self.fade_out))
            alpha = a_in * a_out
            if alpha <= self.min_alpha:
                continue
            x1, y1, x2, y2 = t.xyxy()
            out.append(SBox(tid, x1, y1, x2, y2, t.conf, alpha, t.cls, t.label))
        out.sort(key=lambda b: b.id)
        return out
