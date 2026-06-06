#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
publish_dataset.py — 自我演化检测系统 · 第 1 步
SelfEvolvingRecognition 标注  ->  质量过滤 / 选择性发布  ->  导出 YOLO 训练格式

两个子命令:
  scan    扫描标注目录,读每个 shape 的 score 与几何信息,按置信度+质量给每张图
          打三档分诊(keep / review / reject),写出 manifest.csv。
            keep   = 干净、可直接自动发布;
            review = 边界/可疑/含 difficult/含人工无分框,优先送你人工复核(学习价值最高);
            reject = 没有可用框,丢弃。
  export  读 manifest.csv,只导出 publish=1 的图,写成 ultralytics 可直接 `yolo train` 的数据集。
          类别表 classes.txt 与验证集 frozen_val.json 一旦建立即冻结/只追加 ——
          这样验证集是个跨训练轮次不变的尺子,第 2 步的 promote/回退才有客观依据。

设计原则:
- 完全对齐 SelfEvolvingRecognition 的真实 JSON(version/shapes[label,score,points,shape_type,difficult]/imageWidth/imageHeight)。
- 归一化与其 label_converter.custom_to_yolo 的 hbb 模式逐位一致(对角点 -> xywh 归一化)。
- score 为 None(人工/打标签场景)按"人工放置=可信"兜底,不因此误杀。
- 只依赖标准库;Pillow 仅在 --verify-images 时用到,没装也能跑。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import os.path as osp
import random
import shutil
import sys
from dataclasses import dataclass, field
from glob import glob
from typing import Optional


# --------------------------------------------------------------------------------------
# 配置:质量阈值(全部可调,默认值偏保守)
# --------------------------------------------------------------------------------------
@dataclass
class QualityConfig:
    score_keep: float = 0.50        # 框置信度低于此值 -> 不进训练(视为不可靠伪标签)
    score_review_hi: float = 0.70   # 保留框里只要有 [score_keep, score_review_hi) 的 -> 该图送复核
    min_area_frac: float = 0.0005   # 框面积占比小于此值 -> 视为退化/噪声框,剔除
    keep_unscored: bool = True      # score=None(人工框)是否保留(默认是,人工放置=可信)
    drop_difficult: bool = False    # 是否直接剔除标了 difficult 的框(默认否,只用它触发复核)
    allowed_shape_types: tuple = ("rectangle",)  # 第 1 步聚焦检测框;polygon 按外接矩形处理


# --------------------------------------------------------------------------------------
# 几何工具
# --------------------------------------------------------------------------------------
def _bbox_from_points(points):
    """支持 2 点(对角,旧格式)/ 4 点(新格式)/ 多边形,统一取外接矩形。"""
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _to_yolo_xywh(xmin, ymin, xmax, ymax, w, h):
    """对角点 -> 归一化 xywh,与 custom_to_yolo 的 hbb 一致,并 clamp 到 [0,1]。"""
    xc = (xmin + xmax) / (2.0 * w)
    yc = (ymin + ymax) / (2.0 * h)
    bw = abs(xmax - xmin) / w
    bh = abs(ymax - ymin) / h
    clamp = lambda v: max(0.0, min(1.0, v))
    return clamp(xc), clamp(yc), clamp(bw), clamp(bh)


# --------------------------------------------------------------------------------------
# 单张图的扫描结果
# --------------------------------------------------------------------------------------
@dataclass
class ImageRecord:
    json_path: str
    image_path: str            # 解析后的图片绝对路径("" 表示找不到)
    width: int
    height: int
    n_boxes: int = 0           # 原始矩形/多边形框总数
    n_kept: int = 0            # 通过质量过滤、会进训练的框数
    n_unscored: int = 0        # 保留框里 score 为 None 的数量
    n_difficult: int = 0       # 保留框里标了 difficult 的数量
    n_degenerate: int = 0      # 因面积过小被剔除的框数
    n_skipped_type: int = 0    # 因 shape_type 不支持被跳过的数量
    min_score: Optional[float] = None
    mean_score: Optional[float] = None
    labels: set = field(default_factory=set)
    verdict: str = "reject"
    kept_lines: list = field(default_factory=list)  # 已生成的 YOLO 行(类名占位,导出时换索引)


def _resolve_image(json_path: str, image_path_field: str, images_dir: Optional[str]) -> str:
    """优先 JSON 同目录,其次显式 images_dir,按文件名匹配。"""
    base = osp.basename(image_path_field) if image_path_field else (
        osp.splitext(osp.basename(json_path))[0] + ".jpg"
    )
    cand = osp.join(osp.dirname(json_path), base)
    if osp.exists(cand):
        return osp.abspath(cand)
    if images_dir:
        cand2 = osp.join(images_dir, base)
        if osp.exists(cand2):
            return osp.abspath(cand2)
        # 同名不同后缀兜底
        stem = osp.splitext(base)[0]
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            c = osp.join(images_dir, stem + ext)
            if osp.exists(c):
                return osp.abspath(c)
    return ""


def scan_one(json_path: str, cfg: QualityConfig, images_dir: Optional[str],
             classes_filter: Optional[set]) -> ImageRecord:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    W = int(data.get("imageWidth") or 0)
    H = int(data.get("imageHeight") or 0)
    rec = ImageRecord(
        json_path=osp.abspath(json_path),
        image_path=_resolve_image(json_path, data.get("imagePath", ""), images_dir),
        width=W, height=H,
    )
    if W <= 0 or H <= 0:
        rec.verdict = "reject"  # 没有有效尺寸无法归一化
        return rec

    scored_kept = []
    for shape in data.get("shapes", []):
        st = shape.get("shape_type")
        if st not in cfg.allowed_shape_types and st != "polygon":
            rec.n_skipped_type += 1
            continue
        pts = shape.get("points") or []
        if len(pts) < 2:
            rec.n_skipped_type += 1
            continue
        rec.n_boxes += 1

        label = shape.get("label")
        if classes_filter is not None and label not in classes_filter:
            continue  # 不在目标类里,既不计入也不影响 verdict

        xmin, ymin, xmax, ymax = _bbox_from_points(pts)
        xc, yc, bw, bh = _to_yolo_xywh(xmin, ymin, xmax, ymax, W, H)
        area_frac = bw * bh
        if area_frac < cfg.min_area_frac or bw <= 0 or bh <= 0:
            rec.n_degenerate += 1
            continue

        score = shape.get("score", None)
        score = float(score) if isinstance(score, (int, float)) else None
        if score is None:
            if not cfg.keep_unscored:
                continue
        else:
            if score < cfg.score_keep:
                continue  # 低置信伪标签:不进训练

        difficult = bool(shape.get("difficult", False))
        if difficult and cfg.drop_difficult:
            continue

        # 该框保留
        rec.n_kept += 1
        rec.labels.add(label)
        if score is None:
            rec.n_unscored += 1
        else:
            scored_kept.append(score)
        if difficult:
            rec.n_difficult += 1
        rec.kept_lines.append((label, xc, yc, bw, bh))

    if scored_kept:
        rec.min_score = round(min(scored_kept), 4)
        rec.mean_score = round(sum(scored_kept) / len(scored_kept), 4)

    # ---- 三档分诊 ----
    if rec.n_kept == 0:
        rec.verdict = "reject"
    else:
        borderline = (rec.min_score is not None and rec.min_score < cfg.score_review_hi)
        mixed_provenance = (rec.n_unscored > 0 and len(scored_kept) > 0)
        if borderline or rec.n_difficult > 0 or mixed_provenance:
            rec.verdict = "review"
        else:
            rec.verdict = "keep"
    return rec


def scan_paths(json_files, cfg, images_dir=None, classes_filter=None):
    """库接口:扫描一批标注 JSON,返回 ImageRecord 列表(GUI 与 CLI 共用)。"""
    records = []
    for jp in json_files:
        try:
            records.append(scan_one(jp, cfg, images_dir, classes_filter))
        except Exception as e:
            print(f"[scan] 跳过损坏标注 {jp}: {e}", file=sys.stderr)
    return records


# --------------------------------------------------------------------------------------
# scan 子命令
# --------------------------------------------------------------------------------------
MANIFEST_FIELDS = [
    "image_path", "json_path", "width", "height",
    "n_boxes", "n_kept", "n_unscored", "n_difficult", "n_degenerate", "n_skipped_type",
    "min_score", "mean_score", "labels", "auto_verdict", "publish",
]


def cmd_scan(args):
    cfg = QualityConfig(
        score_keep=args.score_keep,
        score_review_hi=args.score_review_hi,
        min_area_frac=args.min_area_frac,
        keep_unscored=not args.drop_unscored,
        drop_difficult=args.drop_difficult,
    )
    classes_filter = set(args.classes.split(",")) if args.classes else None

    json_files = sorted(glob(osp.join(args.ann_dir, "**", "*.json"), recursive=True))
    if not json_files:
        print(f"[scan] 在 {args.ann_dir} 下没找到任何 .json 标注", file=sys.stderr)
        return 1

    counts = {"keep": 0, "review": 0, "reject": 0}
    rows = []
    for rec in scan_paths(json_files, cfg, args.images_dir, classes_filter):
        counts[rec.verdict] += 1
        rows.append({
            "image_path": rec.image_path,
            "json_path": rec.json_path,
            "width": rec.width, "height": rec.height,
            "n_boxes": rec.n_boxes, "n_kept": rec.n_kept,
            "n_unscored": rec.n_unscored, "n_difficult": rec.n_difficult,
            "n_degenerate": rec.n_degenerate, "n_skipped_type": rec.n_skipped_type,
            "min_score": rec.min_score if rec.min_score is not None else "",
            "mean_score": rec.mean_score if rec.mean_score is not None else "",
            "labels": "|".join(sorted(rec.labels)),
            "auto_verdict": rec.verdict,
            # 默认只把 keep 标为待发布;review/reject 默认不发,由你在 GUI/表格里改 publish 列
            "publish": 1 if rec.verdict == "keep" else 0,
        })

    with open(args.manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)

    total = len(rows)
    print(f"[scan] 共 {total} 张  ->  keep={counts['keep']}  review={counts['review']}  reject={counts['reject']}")
    print(f"[scan] manifest 写到: {args.manifest}")
    print("[scan] 下一步:在 GUI/表格里复核 review 的图,把要发布的行 publish 改成 1,再跑 export")
    return 0


# --------------------------------------------------------------------------------------
# 冻结资源:类别表(只追加,索引稳定) + 验证集(冻结,只追加)
# --------------------------------------------------------------------------------------
def load_or_init_classes(dataset_dir: str, discovered: list) -> list:
    path = osp.join(dataset_dir, "classes.txt")
    classes = []
    if osp.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            classes = [ln.strip() for ln in f if ln.strip()]
    # 新出现的类追加到末尾(已有索引绝不变动)
    for c in discovered:
        if c not in classes:
            classes.append(c)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(classes) + "\n")
    return classes


def load_frozen_val(dataset_dir: str) -> set:
    path = osp.join(dataset_dir, "frozen_val.json")
    if osp.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return set(json.load(f).get("val_stems", []))
    return set()


def save_frozen_val(dataset_dir: str, val_stems: set):
    path = osp.join(dataset_dir, "frozen_val.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"val_stems": sorted(val_stems)}, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------------------
# export 子命令
# --------------------------------------------------------------------------------------
def _verify_image(path: str) -> bool:
    try:
        from PIL import Image  # 可选依赖
    except Exception:
        return True  # 没装 Pillow 就跳过校验
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def export_records(records, out, val_ratio=0.2, seed=42, link=False,
                   grow_val=False, verify_images=False, progress_cb=None,
                   per_class_val=0):
    """库接口:把一批已选定发布的 ImageRecord 导出成 YOLO 数据集。
    classes.txt / frozen_val.json 一旦建立即冻结/只追加;返回汇总 dict。
    GUI 与 CLI 共用此函数,保证导出口径完全一致。
    progress_cb(done:int, total:int, stem:str) 可选,用于 GUI 进度条。"""
    per_image = []   # (stem, image_path, kept_lines)
    discovered = []
    for rec in records:
        if rec.n_kept == 0 or not rec.image_path:
            continue
        if verify_images and not _verify_image(rec.image_path):
            print(f"[export] 跳过(图片损坏): {rec.image_path}", file=sys.stderr)
            continue
        stem = osp.splitext(osp.basename(rec.image_path))[0]
        for lab in rec.labels:
            if lab not in discovered:
                discovered.append(lab)
        per_image.append((stem, rec.image_path, rec.kept_lines))

    if not per_image:
        return {"ok": False, "msg": "过滤后没有可用图片", "n_train": 0, "n_val": 0}

    os.makedirs(out, exist_ok=True)
    classes = load_or_init_classes(out, sorted(discovered))
    cls_index = {c: i for i, c in enumerate(classes)}

    # 冻结验证集:已有的保持在 val;新图默认进 train(val 只增不减、不重洗)
    frozen_val = load_frozen_val(out)
    existing_stems = {s for (s, _, _) in per_image}
    new_stems = [s for (s, _, _) in per_image if s not in frozen_val]

    # 预先算 stem -> 类名集合,以及每个类的图片数(用于保护单样本类不被掏进 val)
    stem_labels = {}
    for (stem, _img, kept) in per_image:
        stem_labels[stem] = {row[0] for row in kept}
    from collections import Counter
    class_img_cnt = Counter()
    for labs in stem_labels.values():
        for lab in labs:
            class_img_cnt[lab] += 1

    def _safe_for_val(stem):
        # 若该图含有"全局只有 1 张图的类",放进 val 会让 train 失去该类 -> 不安全
        return all(class_img_cnt[lab] > 1 for lab in stem_labels.get(stem, ()))

    if not frozen_val:
        rng = random.Random(seed)
        k = max(1, int(round(len(new_stems) * val_ratio)))
        rng.shuffle(new_stems)
        # 优先从"安全"的图里选 val,避免把单样本类掏空
        safe = [s for s in new_stems if _safe_for_val(s)]
        frozen_val = set(safe[:k])
    elif grow_val:
        rng = random.Random(seed)
        rng.shuffle(new_stems)
        target_val = int(round(len(existing_stems) * val_ratio))
        need = max(0, target_val - len(frozen_val & existing_stems))
        safe = [s for s in new_stems if _safe_for_val(s)]
        frozen_val |= set(safe[:need])

    # 按类保证:每个类至少 per_class_val 个实例进 val(小数据集上让每类都有验证样本,
    # 否则 bed/umbrella 这种单样本类永远进不了 val,mAP 只能在 0/0.33/0.66/1 间跳)。
    # 注意:只有图片数 > per_class_val 的类才可能满足(单样本类优先留 train,无法验证)。
    if per_class_val and per_class_val > 0:
        val_class_cnt = Counter()
        for s in frozen_val:
            for lab in stem_labels.get(s, ()):
                val_class_cnt[lab] += 1
        rng = random.Random(seed + 1)
        unverifiable = []
        for lab in sorted(class_img_cnt):
            if class_img_cnt[lab] <= 1:
                unverifiable.append(lab)        # 全局仅 1 张图,无法既训练又验证
                continue
            cands = [s for s in stem_labels
                     if lab in stem_labels[s] and s not in frozen_val]
            rng.shuffle(cands)
            for s in cands:
                if val_class_cnt[lab] >= per_class_val:
                    break
                # 把 s 放进 val 后,确保该图涉及的每个类在 train 里都还剩至少 1 张
                if not all(
                    class_img_cnt[l2] - (1 + sum(1 for vs in frozen_val if l2 in stem_labels.get(vs, ()))) >= 1
                    for l2 in stem_labels[s]
                ):
                    continue
                frozen_val.add(s)
                for l2 in stem_labels[s]:
                    val_class_cnt[l2] += 1
        if unverifiable:
            print(f"[export] 提示:这些类只有 1 张图,无法进验证集(已留在训练集): {', '.join(unverifiable)}",
                  file=sys.stderr)
    save_frozen_val(out, frozen_val)

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(osp.join(out, sub), exist_ok=True)

    n_train = n_val = 0
    total = len(per_image)
    for i, (stem, img_path, kept_lines) in enumerate(per_image, 1):
        split = "val" if stem in frozen_val else "train"
        ext = osp.splitext(img_path)[1] or ".jpg"
        dst_img = osp.join(out, "images", split, stem + ext)
        try:
            if link:
                if osp.lexists(dst_img):
                    os.remove(dst_img)
                os.symlink(img_path, dst_img)
            else:
                shutil.copy2(img_path, dst_img)
        except Exception as e:
            print(f"[export] 复制失败 {img_path}: {e}", file=sys.stderr)
            continue
        with open(osp.join(out, "labels", split, stem + ".txt"), "w", encoding="utf-8") as f:
            for lab, xc, yc, bw, bh in kept_lines:
                if lab not in cls_index:
                    continue
                f.write(f"{cls_index[lab]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        if split == "val":
            n_val += 1
        else:
            n_train += 1
        if progress_cb:
            progress_cb(i, total, stem)

    names_block = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
    yaml_text = (
        f"# 由 publish_dataset.py 生成 —— 类别表与 val 已冻结/只追加\n"
        f"path: {osp.abspath(out)}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n{names_block}\n"
    )
    with open(osp.join(out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml_text)

    return {"ok": True, "out": osp.abspath(out), "n_train": n_train, "n_val": n_val,
            "n_classes": len(classes), "n_val_frozen": len(frozen_val), "classes": classes}


def cmd_export(args):
    # 读 manifest,取 publish=1
    rows = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if str(r.get("publish", "0")).strip() in ("1", "true", "True"):
                rows.append(r)
    if not rows:
        print("[export] manifest 里没有 publish=1 的行,无可导出", file=sys.stderr)
        return 1

    cfg = QualityConfig()  # 与分诊同一套阈值重算保留框,口径一致
    classes_filter = set(args.classes.split(",")) if args.classes else None
    recs = []
    for r in rows:
        jp = r["json_path"]
        if not osp.exists(jp):
            print(f"[export] 跳过(标注不存在): {jp}", file=sys.stderr)
            continue
        recs.append(scan_one(jp, cfg, args.images_dir, classes_filter))

    s = export_records(recs, args.out, val_ratio=args.val_ratio, seed=args.seed,
                       link=args.link, grow_val=args.grow_val,
                       verify_images=args.verify_images,
                       per_class_val=args.per_class_val)
    if not s.get("ok"):
        print(f"[export] {s.get('msg', '导出失败')}", file=sys.stderr)
        return 1
    print(f"[export] 导出完成:train={s['n_train']}  val={s['n_val']}  类别数={s['n_classes']}")
    print(f"[export] 数据集目录: {s['out']}")
    print(f"[export] 冻结资源: classes.txt({s['n_classes']} 类), frozen_val.json({s['n_val_frozen']} 张)")
    print(f"[export] 训练即可: yolo detect train data={osp.join(s['out'], 'data.yaml')} model=yolo11n.pt")
    return 0


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="标注结果 -> 质量过滤/选择性发布 -> YOLO 数据集")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="扫描标注并生成 manifest.csv(含三档分诊)")
    s.add_argument("ann_dir", help="标注 .json 所在目录(递归)")
    s.add_argument("--images-dir", default=None, help="图片目录(默认与 json 同目录)")
    s.add_argument("--manifest", default="manifest.csv")
    s.add_argument("--classes", default=None, help="逗号分隔的目标类(只保留这些类;留空=全部)")
    s.add_argument("--score-keep", type=float, default=0.50)
    s.add_argument("--score-review-hi", type=float, default=0.70)
    s.add_argument("--min-area-frac", type=float, default=0.0005)
    s.add_argument("--drop-unscored", action="store_true", help="剔除 score 为 None 的人工框")
    s.add_argument("--drop-difficult", action="store_true", help="直接剔除 difficult 框")
    s.set_defaults(func=cmd_scan)

    e = sub.add_parser("export", help="按 manifest 的 publish 列导出 YOLO 数据集")
    e.add_argument("manifest", help="scan 生成的 manifest.csv(已复核/改过 publish)")
    e.add_argument("--out", required=True, help="输出数据集目录")
    e.add_argument("--images-dir", default=None)
    e.add_argument("--classes", default=None, help="目标类过滤(应与 scan 时一致)")
    e.add_argument("--val-ratio", type=float, default=0.2)
    e.add_argument("--grow-val", action="store_true", help="允许用新图按比例补充冻结 val(只增不减)")
    e.add_argument("--per-class-val", type=int, default=0,
                   help="保证每个类至少 N 个实例进 val(小数据集建议 1-2;0=关闭)")
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--link", action="store_true", help="用软链接代替复制图片(省磁盘)")
    e.add_argument("--verify-images", action="store_true", help="导出前用 Pillow 校验图片可读")
    e.set_defaults(func=cmd_export)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
