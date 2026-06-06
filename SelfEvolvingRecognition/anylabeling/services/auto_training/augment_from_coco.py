# -*- coding: utf-8 -*-
"""
augment_from_coco.py — 从 COCO 格式数据集(如本地 coco128)按类名注入你的数据集

目的:你的某些类样本太少(bed/umbrella 各 1 张)。coco128 里有大量 person/car/bed/umbrella 等
COCO 类,本脚本把其中"你关心的类"的图按【类名】映射到你的 classes.txt 索引,追加进 train/val。

特点:
  - 只注入你 classes.txt 里【已有】的类(按名字匹配 COCO 类名),其余 COCO 类的框丢弃。
  - 一张 COCO 图若含多个类,只保留你有的那些类的框;若过滤后没框则跳过该图。
  - 追加模式:不动你现有的图与标注;新图加前缀避免重名。
  - 按比例分到 train/val(val 同样可保证每类至少 N 个)。
  - 预演(默认)只报告;--apply 才写。

用法:
  python augment_from_coco.py --dataset D:\\code\\yolo\\datasets --coco D:\\code\\yolo\\datasets\\coco128
  # 只注入指定类:
  python augment_from_coco.py --dataset ... --coco ... --classes person,car,bed,umbrella
  python augment_from_coco.py --dataset ... --coco ... --apply --val-ratio 0.2 --per-class-val 2
"""
import argparse
import os
import os.path as osp
import random
import shutil
from collections import Counter

try:
    from anylabeling.services.auto_training import replay as R
except ImportError:
    import replay as R

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_classes(path):
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def find_coco_split_dirs(coco_root):
    """返回 [(images_dir, labels_dir), ...]。兼容 coco128/images/train2017 与 images/train 等。"""
    pairs = []
    for root, _dirs, files in os.walk(coco_root):
        if osp.basename(root) == "images" or "/images/" in root.replace("\\", "/") + "/":
            # 找对应 labels
            lbl = root.replace("images", "labels", 1)
            if osp.isdir(lbl):
                # 同时有图和标注
                has_img = any(osp.splitext(f)[1].lower() in IMG_EXTS for f in files)
                if has_img:
                    pairs.append((root, lbl))
    # 去重
    uniq = []
    seen = set()
    for a, b in pairs:
        if a not in seen:
            uniq.append((a, b))
            seen.add(a)
    return uniq


def scan_coco(coco_root, coco_names, want_names):
    """扫描 coco 图片,返回 [(img_path, [(your_name, line_floats...)...]), ...](已按 want_names 过滤)。"""
    want = set(want_names)
    name_by_id = {i: n for i, n in enumerate(coco_names)}
    out = []
    for img_dir, lbl_dir in find_coco_split_dirs(coco_root):
        for fn in sorted(os.listdir(img_dir)):
            if osp.splitext(fn)[1].lower() not in IMG_EXTS:
                continue
            stem = osp.splitext(fn)[0]
            lbl = osp.join(lbl_dir, stem + ".txt")
            if not osp.exists(lbl):
                continue
            kept = []
            for ln in open(lbl, encoding="utf-8"):
                s = ln.split()
                if len(s) < 5:
                    continue
                cid = int(float(s[0]))
                nm = name_by_id.get(cid)
                if nm in want:
                    kept.append((nm, s[1], s[2], s[3], s[4]))
            if kept:
                out.append((osp.join(img_dir, fn), kept))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="你的 YOLO 数据集目录(含 classes.txt)")
    ap.add_argument("--coco", required=True, help="COCO 格式数据集根目录(如 coco128)")
    ap.add_argument("--classes", default=None,
                    help="只注入这些类(逗号分隔);默认=你 classes.txt 与 COCO 的交集")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--per-class-val", type=int, default=1,
                    help="保证每个被注入类至少 N 个新图进 val")
    ap.add_argument("--max-per-class", type=int, default=0,
                    help="每类最多注入多少张图(0=不限)")
    ap.add_argument("--prefix", default="coco_", help="新图文件名前缀(避免与你现有图重名)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--apply", action="store_true", help="真正写入(默认仅预演)")
    a = ap.parse_args(argv)

    cls_path = osp.join(a.dataset, "classes.txt")
    your_classes = read_classes(cls_path)
    your_idx = {c: i for i, c in enumerate(your_classes)}
    coco_names = R.COCO80_NAMES

    # 决定注入哪些类
    if a.classes:
        want = [c.strip() for c in a.classes.split(",") if c.strip()]
        bad = [c for c in want if c not in your_idx]
        if bad:
            print(f"[augment] 警告:这些类不在你的 classes.txt,将忽略: {', '.join(bad)}")
        bad2 = [c for c in want if c not in coco_names]
        if bad2:
            print(f"[augment] 警告:这些类不是 COCO 类名,coco128 里没有: {', '.join(bad2)}")
        want = [c for c in want if c in your_idx and c in coco_names]
    else:
        want = [c for c in your_classes if c in coco_names]
    if not want:
        print("[augment] 没有可注入的类(你的 classes.txt 与 COCO 类名无交集,或指定类无效)")
        return 1
    print(f"[augment] 将注入这些类(你的索引): " + ", ".join(f"{c}#{your_idx[c]}" for c in want))

    found = scan_coco(a.coco, coco_names, want)
    print(f"[augment] 在 COCO 数据集中找到 {len(found)} 张含目标类的图")
    if not found:
        return 1

    # 每类计数 + max-per-class 限制 + 选图
    rng = random.Random(a.seed)
    rng.shuffle(found)
    per_class = Counter()
    chosen = []
    for img, kept in found:
        labs = {k[0] for k in kept}
        if a.max_per_class:
            # 若该图的所有类都已达上限,跳过
            if all(per_class[l] >= a.max_per_class for l in labs):
                continue
        chosen.append((img, kept))
        for l in labs:
            per_class[l] += 1
    print(f"[augment] 选中 {len(chosen)} 张注入")
    for c in want:
        print(f"   {c:12s} 注入图数≈{per_class[c]}")

    # 分 train/val:每类保证 per_class_val 进 val
    stem_labels = {}
    for i, (img, kept) in enumerate(chosen):
        stem_labels[i] = {k[0] for k in kept}
    val_idx = set()
    val_class_cnt = Counter()
    # 先按比例
    k = int(round(len(chosen) * a.val_ratio))
    order = list(range(len(chosen)))
    rng.shuffle(order)
    for i in order[:k]:
        val_idx.add(i)
        for l in stem_labels[i]:
            val_class_cnt[l] += 1
    # 再保证每类
    if a.per_class_val > 0:
        for c in want:
            if val_class_cnt[c] >= a.per_class_val:
                continue
            for i in order:
                if i in val_idx:
                    continue
                if c in stem_labels[i]:
                    val_idx.add(i)
                    for l in stem_labels[i]:
                        val_class_cnt[l] += 1
                    if val_class_cnt[c] >= a.per_class_val:
                        break

    n_train = len(chosen) - len(val_idx)
    print(f"[augment] 分流:train +{n_train} 张, val +{len(val_idx)} 张")

    if not a.apply:
        print("\n[预演] 未写入。确认后加 --apply 执行。")
        return 0

    # 写入(追加),图名加前缀避免重名
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(osp.join(a.dataset, sub), exist_ok=True)
    written = 0
    for i, (img, kept) in enumerate(chosen):
        split = "val" if i in val_idx else "train"
        ext = osp.splitext(img)[1].lower()
        base = a.prefix + osp.splitext(osp.basename(img))[0]
        shutil.copy2(img, osp.join(a.dataset, "images", split, base + ext))
        with open(osp.join(a.dataset, "labels", split, base + ".txt"), "w", encoding="utf-8") as f:
            for (nm, xc, yc, bw, bh) in kept:
                f.write(f"{your_idx[nm]} {xc} {yc} {bw} {bh}\n")
        written += 1
    print(f"[augment] 已写入 {written} 张图(train {n_train} / val {len(val_idx)})到 {a.dataset}")
    print(f"[augment] 重新训练即可让这些类样本变多:Model Registry -> 训练一轮")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
