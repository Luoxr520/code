# -*- coding: utf-8 -*-
"""
collector.py — 自我演化系统 · 数据采集(闭环的"持续采集"一环)

从视频流(视频文件/摄像头)按策略间隔采集帧,存成按时间命名的文件夹,作为步骤一(自动标注)的输入。
采集策略:
  - 变化触发:仅当当前帧与上一张【已存帧】差异够大时才存(避免狂存几乎一样的废帧)。
  - 检测触发(可选):仅当模型在该帧检测到目标时才存(只采"有价值"的帧)。
  - 最短间隔:两次存帧至少间隔 min_gap 秒(防止 1 秒存几十张)。
  - 兜底间隔:即使没变化,每隔 max_gap 秒也强制存一张(保证长时间静止场景也有数据)。

本文件的核心判定逻辑是纯函数(frame_changed / should_capture / session_dirname / frame_filename),
可单测;cv2 读帧、模型推理在本机运行。
"""
import os
import os.path as osp
import time
import json

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------- 纯函数(可测)
def session_dirname(ts=None):
    """采集会话目录名:按开始时间。"""
    t = time.localtime(ts if ts is not None else time.time())
    return time.strftime("%Y-%m-%d_%H-%M-%S", t)


def frame_filename(index, ts=None):
    """单帧文件名:帧序号 + 时刻(便于排序与溯源)。"""
    t = time.localtime(ts if ts is not None else time.time())
    return f"frame_{index:06d}_{time.strftime('%H%M%S', t)}.jpg"


def frame_diff_score(gray_prev, gray_cur):
    """两帧(单通道、同尺寸)的归一化差异分 [0,1]:平均绝对差 / 255。
    传入应为 numpy 二维数组(用 cv2 转灰度+缩放后的小图,省算力)。"""
    import numpy as np

    if gray_prev is None or gray_cur is None:
        return 1.0  # 没有上一帧 -> 视为最大变化(第一帧总采)
    a = gray_prev.astype("float32")
    b = gray_cur.astype("float32")
    return float(np.abs(a - b).mean() / 255.0)


def should_capture(diff_score, has_detection, *, change_thresh,
                   use_detection, secs_since_last, min_gap, max_gap):
    """综合判定这一帧是否应该存。返回 (是否存, 原因字符串)。
    规则优先级:
      1) 距上次存帧不足 min_gap 秒 -> 一律不存(去抖)。
      2) 超过 max_gap 秒没存 -> 兜底强制存。
      3) 检测触发开:有检测目标 且 变化达标 -> 存。
      4) 仅变化触发:变化达标 -> 存。
    """
    if secs_since_last < min_gap:
        return False, "min_gap"
    if max_gap and secs_since_last >= max_gap:
        return True, "max_gap"
    changed = diff_score >= change_thresh
    if use_detection:
        if has_detection and changed:
            return True, "detect+change"
        if has_detection and not changed:
            return False, "detect_but_static"
        return False, "no_detection"
    # 仅变化触发
    if changed:
        return True, "change"
    return False, "static"


# ---------------------------------------------------------------- 会话写入
class CaptureSession:
    """管理一次采集会话:建时间文件夹、存帧、写 session.json。"""

    def __init__(self, root_dir, source_desc="", strategy=None, start_ts=None):
        self.root = root_dir
        self.name = session_dirname(start_ts)
        self.dir = osp.join(root_dir, self.name)
        os.makedirs(self.dir, exist_ok=True)
        self.count = 0
        self.start_ts = start_ts if start_ts is not None else time.time()
        self.meta = {
            "session": self.name,
            "source": source_desc,
            "strategy": strategy or {},
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_ts)),
            "frames": [],
        }

    def save_frame(self, frame_bgr, ts=None, reason="", writer=None):
        """存一帧(frame_bgr 为 BGR ndarray)。writer 用于注入(测试时可替换 cv2.imwrite)。"""
        ts = ts if ts is not None else time.time()
        self.count += 1
        fn = frame_filename(self.count, ts)
        path = osp.join(self.dir, fn)
        if writer is None:
            import cv2
            cv2.imwrite(path, frame_bgr)
        else:
            writer(path, frame_bgr)
        self.meta["frames"].append({
            "file": fn,
            "index": self.count,
            "ts": time.strftime("%H:%M:%S", time.localtime(ts)),
            "reason": reason,
        })
        return path

    def finalize(self):
        """写 session.json,返回会话目录。"""
        self.meta["frame_count"] = self.count
        self.meta["ended_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(osp.join(self.dir, "session.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)
        return self.dir


# ---------------------------------------------------------------- 采集 -> 自动标注桥接
def _xyxy_to_points(x1, y1, x2, y2):
    """矩形两对角点 -> SelfEvolvingRecognition 的 points(左上、右下)。"""
    return [[float(x1), float(y1)], [float(x2), float(y2)]]


def build_anylabeling_json(image_filename, width, height, dets, names):
    """把检测结果组装成 SelfEvolvingRecognition 的标注 dict。
    dets: [(x1,y1,x2,y2,conf,cls), ...](绝对像素);names: 类名列表/字典(cls->name)。
    纯函数,可单测(对齐 SelfEvolvingRecognition 真实 JSON 字段)。"""
    def name_of(c):
        if isinstance(names, dict):
            return names.get(int(c), str(int(c)))
        try:
            return names[int(c)]
        except (IndexError, TypeError):
            return str(int(c))

    shapes = []
    for (x1, y1, x2, y2, conf, cls) in dets:
        shapes.append({
            "label": name_of(cls),
            "score": float(conf),
            "points": _xyxy_to_points(x1, y1, x2, y2),
            "group_id": None,
            "difficult": False,
            "shape_type": "rectangle",
            "flags": {},
            "attributes": {},
        })
    return {
        "version": "2.4.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": int(height),
        "imageWidth": int(width),
        "description": "auto-labeled by evo current model",
    }


def autolabel_dir(image_dir, predictor, names, *, conf=0.25, writer=None):
    """对 image_dir 里每张图用 predictor 预测,写同名 .json(SelfEvolvingRecognition 可直接打开)。
    predictor(path)-> (width, height, dets);dets 为绝对像素 [(x1,y1,x2,y2,conf,cls)...]。
    writer(json_path, dict) 可注入(测试用);返回 (图片数, 标注框总数)。"""
    if writer is None:
        def writer(p, d):
            with open(p, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)

    imgs = sorted(
        f for f in os.listdir(image_dir)
        if osp.splitext(f)[1].lower() in IMG_EXTS
    )
    n_img = n_box = 0
    for fn in imgs:
        path = osp.join(image_dir, fn)
        w, h, dets = predictor(path)
        dets = [d for d in dets if float(d[4]) >= conf]
        data = build_anylabeling_json(fn, w, h, dets, names)
        writer(osp.join(image_dir, osp.splitext(fn)[0] + ".json"), data)
        n_img += 1
        n_box += len(dets)
    return n_img, n_box
