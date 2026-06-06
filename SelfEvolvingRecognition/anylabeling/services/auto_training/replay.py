# -*- coding: utf-8 -*-
"""
replay.py — 自我演化检测系统 · 抗遗忘:经验回放数据组装

训练时把外部/历史样本(尤其 coco128)按"类名"重映射到你当前的类别空间,
混进训练集一起训;val 仍用冻结验证集。这样小数据持续训练不会把通用识别能力忘掉。

核心是纯数据处理(重映射 / 配对发现 / 合成训练目录),不依赖 ultralytics,便于测试。
真正的训练在你机器上由 train_with_registry 调 ultralytics 完成。
"""
from __future__ import annotations

import os
import os.path as osp
import random
import shutil

import yaml

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# COCO80 类名(coco128 等没带 names 时的兜底)
COCO80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


def _norm(p):
    return str(p).replace("\\", "/")


def _img2label(img):
    """把图片路径里最后一个 images 目录换成 labels,扩展名换 .txt(ultralytics 约定)。"""
    parts = _norm(img).split("/")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    base, _ = osp.splitext("/".join(parts))
    return base + ".txt"


def yolo_image_label_pairs(root):
    """在一个 YOLO 风格目录下找 (图片, 标签) 配对(只保留标签存在的)。"""
    root = osp.abspath(root)
    pairs = []
    for dp, _, files in os.walk(root):
        if (os.sep + "labels" + os.sep) in (dp + os.sep) or dp.endswith(os.sep + "labels"):
            continue  # 跳过 labels 目录本身
        for fn in files:
            if osp.splitext(fn)[1].lower() in IMG_EXTS:
                img = osp.join(dp, fn)
                lbl = _img2label(img)
                if osp.exists(lbl):
                    pairs.append((img, lbl))
    pairs.sort()
    return pairs


def load_class_names(root):
    """从目录里的 data.yaml / *.yaml / classes.txt 读类别名;返回 {idx:name} 或 None。"""
    root = osp.abspath(root)
    # yaml
    cands = [osp.join(root, "data.yaml")] + [
        osp.join(root, f) for f in (os.listdir(root) if osp.isdir(root) else []) if f.endswith(".yaml")
    ]
    for y in cands:
        if osp.exists(y):
            try:
                d = yaml.safe_load(open(y, encoding="utf-8"))
                m = _names_map(d)
                if m:
                    return m
            except Exception:
                pass
    # classes.txt
    ct = osp.join(root, "classes.txt")
    if osp.exists(ct):
        names = [l.strip() for l in open(ct, encoding="utf-8") if l.strip()]
        if names:
            return {i: n for i, n in enumerate(names)}
    return None


def _names_map(d):
    if not isinstance(d, dict):
        return {}
    n = d.get("names")
    if isinstance(n, dict):
        return {int(k): v for k, v in n.items()}
    if isinstance(n, list):
        return {i: v for i, v in enumerate(n)}
    return {}


def remap_label_lines(lines, src_names, dst_name2idx):
    """把一份标签的类别索引从 src 重映射到 dst(按类名);dst 里没有的类别整框丢弃。"""
    out = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 5:
            continue
        try:
            ci = int(float(parts[0]))
        except ValueError:
            continue
        name = src_names.get(ci)
        if name is None:
            continue
        di = dst_name2idx.get(name)
        if di is None:
            continue
        out.append(" ".join([str(di)] + parts[1:]))
    return out


def _coco_fallback(src_root):
    if "coco" in osp.basename(osp.abspath(src_root)).lower():
        return {i: n for i, n in enumerate(COCO80_NAMES)}
    return None


def _resolve_yaml(data_yaml):
    d = yaml.safe_load(open(data_yaml, encoding="utf-8"))
    base = osp.dirname(osp.abspath(data_yaml))
    root = d.get("path")
    root = osp.join(base, str(root)) if (root and not osp.isabs(str(root))) else (str(root) if root else base)

    def res(x):
        x = str(x)
        return x if osp.isabs(x) else osp.join(root, x)

    return d, root, res(d["train"]), res(d.get("val", d["train"]))


def _reset_dir(d):
    if osp.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)


def _copy_pair(img, lbl, imgs_out, lbls_out, stem=None):
    ext = osp.splitext(img)[1]
    stem = stem or osp.splitext(osp.basename(img))[0]
    shutil.copy(img, osp.join(imgs_out, stem + ext))
    shutil.copy(lbl, osp.join(lbls_out, stem + ".txt"))


def seed_into_buffer(src_root, dst_dataset, src_names=None, cap=None, seed=0):
    """把外部数据集(如 coco128)重映射到目标类别,采样后存进 <dst>/replay/。返回加入数量。"""
    src_names = src_names or load_class_names(src_root) or _coco_fallback(src_root)
    if not src_names:
        raise RuntimeError(f"读不到源数据集 {src_root} 的类别名(请提供 src_names 或放 data.yaml/classes.txt)")
    dst_names = load_class_names(dst_dataset)
    if not dst_names:
        raise RuntimeError(f"读不到目标数据集 {dst_dataset} 的类别名(需要 classes.txt 或 data.yaml)")
    dst_n2i = {v: k for k, v in dst_names.items()}

    pairs = yolo_image_label_pairs(src_root)
    random.Random(seed).shuffle(pairs)
    if cap:
        pairs = pairs[:cap]
    imgs_out = osp.join(dst_dataset, "replay", "images")
    lbls_out = osp.join(dst_dataset, "replay", "labels")
    os.makedirs(imgs_out, exist_ok=True)
    os.makedirs(lbls_out, exist_ok=True)
    n = 0
    for img, lbl in pairs:
        rl = remap_label_lines(open(lbl, encoding="utf-8").read().splitlines(), src_names, dst_n2i)
        if not rl:
            continue
        stem = "seed_" + osp.splitext(osp.basename(img))[0]
        shutil.copy(img, osp.join(imgs_out, stem + osp.splitext(img)[1]))
        open(osp.join(lbls_out, stem + ".txt"), "w", encoding="utf-8").write("\n".join(rl) + "\n")
        n += 1
    return n


def buffer_dir(dataset_dir):
    return osp.join(dataset_dir, "replay")


def compose_train_dir(dataset_dir, data_yaml, replay_sources, replay_max, work_dir, seed=0):
    """合成训练集:用户 train + 采样的回放样本(按类名重映射);val 保持冻结验证集不变。
    返回 (合成后的 data.yaml 路径, 统计 dict)。replay_sources: 目录列表(coco128 / 其它 / 缓冲)。"""
    d, root, train_dir, _val_dir = _resolve_yaml(data_yaml)
    dst_names = _names_map(d)
    if not dst_names:
        m = load_class_names(dataset_dir)
        dst_names = m or {}
    dst_n2i = {v: k for k, v in dst_names.items()}

    # 收集回放项 (img, 重映射后的标签行)
    rng = random.Random(seed)
    replay_items = []
    for src in replay_sources:
        if not src or not osp.isdir(src):
            continue
        src_names = load_class_names(src) or _coco_fallback(src)
        if not src_names:
            continue
        for img, lbl in yolo_image_label_pairs(src):
            rl = remap_label_lines(open(lbl, encoding="utf-8").read().splitlines(), src_names, dst_n2i)
            if rl:
                replay_items.append((img, rl))
    rng.shuffle(replay_items)
    if replay_max and replay_max > 0:
        replay_items = replay_items[:replay_max]

    imgs_out = osp.join(work_dir, "images", "train")
    lbls_out = osp.join(work_dir, "labels", "train")
    _reset_dir(imgs_out)
    _reset_dir(lbls_out)

    n_user = 0
    for img, lbl in yolo_image_label_pairs(train_dir):
        _copy_pair(img, lbl, imgs_out, lbls_out)
        n_user += 1
    n_replay = 0
    for img, rl in replay_items:
        stem = "rp_%05d_%s" % (n_replay, osp.splitext(osp.basename(img))[0])
        shutil.copy(img, osp.join(imgs_out, stem + osp.splitext(img)[1]))
        open(osp.join(lbls_out, stem + ".txt"), "w", encoding="utf-8").write("\n".join(rl) + "\n")
        n_replay += 1

    merged = dict(d)
    merged["path"] = osp.abspath(dataset_dir)   # val 相对此路径 -> 仍指向冻结验证集
    merged["train"] = osp.abspath(imgs_out)      # 绝对路径覆盖,指向合成的 train
    out_yaml = osp.join(work_dir, "data.yaml")
    os.makedirs(work_dir, exist_ok=True)
    yaml.safe_dump(merged, open(out_yaml, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
    return out_yaml, {"user": n_user, "replay": n_replay}


# ======================================================================================
# LwF 蒸馏(伪标签实现):老师模型在(无标注)图片上的预测 -> 伪标注,供训练蒸馏
# ======================================================================================
def xyxy_to_yolo_line(cls, x1, y1, x2, y2, w, h):
    """绝对像素 xyxy -> 归一化 YOLO 行 'cls cx cy bw bh'(裁到 [0,1])。"""
    def clamp(v):
        return max(0.0, min(1.0, v))

    cx = clamp(((x1 + x2) / 2.0) / w)
    cy = clamp(((y1 + y2) / 2.0) / h)
    bw = clamp((x2 - x1) / w)
    bh = clamp((y2 - y1) / h)
    return f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def _boxes_to_yolo_lines(boxes, w, h):
    """ultralytics result.boxes -> [YOLO 行,...]。"""
    import numpy as np

    def arr(x):
        return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)

    xyxy, cls = arr(boxes.xyxy), arr(boxes.cls)
    out = []
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = (float(v) for v in xyxy[i][:4])
        out.append(xyxy_to_yolo_line(int(cls[i]), x1, y1, x2, y2, w, h))
    return out


def generate_pseudo_labels(dataset_dir, teacher_ckpt, image_dir, out_dir,
                           conf=0.4, iou=0.45, imgsz=640, device=None, loader=None):
    """老师模型在 image_dir 的图片上预测,写成伪标注到 out_dir(用户类别空间)。
    同时复制 classes.txt,使训练时该目录按用户类别恒等映射。返回 (图片数, 框数)。"""
    if loader is None:
        def loader(c):
            from ultralytics import YOLO
            return YOLO(c)

    model = loader(teacher_ckpt)
    imgs = sorted(
        osp.join(image_dir, f) for f in os.listdir(image_dir)
        if osp.splitext(f)[1].lower() in IMG_EXTS
    )
    imgs_out, lbls_out = osp.join(out_dir, "images"), osp.join(out_dir, "labels")
    _reset_dir(imgs_out)
    _reset_dir(lbls_out)
    src_cls = osp.join(dataset_dir, "classes.txt")
    if osp.exists(src_cls):
        shutil.copy(src_cls, osp.join(out_dir, "classes.txt"))

    n_img = n_box = 0
    for img in imgs:
        res = model.predict(img, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)[0]
        h, w = res.orig_shape
        boxes = getattr(res, "boxes", None)
        lines = _boxes_to_yolo_lines(boxes, w, h) if boxes is not None else []
        stem = osp.splitext(osp.basename(img))[0]
        shutil.copy(img, osp.join(imgs_out, stem + osp.splitext(img)[1]))
        open(osp.join(lbls_out, stem + ".txt"), "w", encoding="utf-8").write(
            "\n".join(lines) + ("\n" if lines else "")
        )
        n_img += 1
        n_box += len(lines)
    return n_img, n_box
