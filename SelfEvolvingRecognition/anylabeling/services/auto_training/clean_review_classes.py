# -*- coding: utf-8 -*-
"""
clean_review_classes.py — 消化「AI 审查标记类」并压缩类别表(安全、可回滚)

背景:标注审查时产生的 AI_REVIEW_* / missing_* 被当成了正式类别,把类别撑大。
这些批注本质是对真实目标的审查意见,应当「消化」回真实类别(或删除),而不是保留为类。

做法:
  1) 备份 labels/ 与 classes.txt、data.yaml 到 <dataset>/_backup_clean_YYYYmmdd_HHMMSS/
  2) 按 REMAP(旧类名 -> 新类名 或 DELETE)处理每个框
  3) 重建干净的 classes.txt(保留出现过的正常类,稳定排序)与 data.yaml(UTF-8)
  4) 重写所有 .txt 的类号到新索引;被删的框丢弃

用法:
  python clean_review_classes.py --dataset D:\\code\\yolo\\datasets            # 预演(只报告,不写)
  python clean_review_classes.py --dataset D:\\code\\yolo\\datasets --apply     # 真正执行
"""
import argparse
import os
import os.path as osp
import shutil
import time

DELETE = "__DELETE__"

# 审查标记类 -> 真实类名 / 删除。键是 classes.txt 里的原始名字。
# 不在此表中的类名 = 正常类,原样保留。
REMAP = {
    "AI_REVIEW_box_adjust_长颈鹿": "giraffe",
    "AI_REVIEW_missing_person": "person",
    "AI_REVIEW_wrong_class_person": "person",
    "AI_REVIEW_wrong_class_stop_sign": "stop sign",
    "missing_car": "car",
    "missing_car_2": "car",
    # 无对应正常类 / 本就是误检 -> 删除这些框
    "AI_REVIEW_false_positive_person": DELETE,
    "AI_REVIEW_false_positive_AI_REVIEW_wrong_class_traffic_sign": DELETE,
    "AI_REVIEW_missing_树木": DELETE,
}

IMG_LABEL_DIRS = [("images/train", "labels/train"), ("images/val", "labels/val")]

# 审查标记前缀:这些都是标注复核流程产生的,不是真实目标类
_REVIEW_PREFIXES = ("AI_REVIEW_", "missing_")
# 审查动作词:从类名里剥掉这些,剩下的可能是真实类名
_REVIEW_ACTIONS = (
    "box_adjust", "false_positive", "wrong_class", "missing",
    "AI_REVIEW", "fp", "fn",
)


def is_review_class(name):
    """是否是审查标记类(应被清理,而不是当真实目标)。"""
    return any(name.startswith(p) for p in _REVIEW_PREFIXES)


def _auto_resolve(name, normal_names):
    """对审查标记类名,先查显式 REMAP;否则尝试从名字尾部提取真实类名:
    若提取出的词在『正常类集合』里 -> 归并到该类;否则 -> 删除。
    normal_names: 数据集里所有非审查类的名字集合(用于判断提取出的词是不是真类)。"""
    if name in REMAP:
        return REMAP[name]
    # 从尾部往前剥审查动作词/前缀,逐步尝试匹配正常类
    tokens = name.replace("AI_REVIEW_", "").split("_")
    # 依次尝试 越来越长的尾部组合,找在正常类里的
    for start in range(len(tokens)):
        cand = "_".join(tokens[start:]).strip()
        if cand and cand in normal_names:
            return cand
        cand_sp = " ".join(tokens[start:]).strip()   # 如 "stop_sign" -> "stop sign"
        if cand_sp and cand_sp in normal_names:
            return cand_sp
    return DELETE   # 提取不出真实类 -> 删除这些框


def read_classes(path):
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def iter_label_files(dataset_dir):
    for _, lbl in IMG_LABEL_DIRS:
        d = osp.join(dataset_dir, lbl)
        if not osp.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.endswith(".txt"):
                yield osp.join(d, fn)


def plan(dataset_dir):
    """计算:旧名表、新名表(压缩后)、旧idx->新idx 或 DELETE。"""
    old = read_classes(osp.join(dataset_dir, "classes.txt"))

    # 正常类集合(非审查标记)= 自动推断归并目标的依据
    normal_names = {n for n in old if not is_review_class(n)}

    # 每个旧类名解析到 目标名 / DELETE / 自身
    resolved = []
    for n in old:
        if is_review_class(n):
            resolved.append(_auto_resolve(n, normal_names))   # 审查类:显式表 or 自动推断
        else:
            resolved.append(n)                                # 正常类:保留

    # 新类表 = 所有「保留下来的目标名」去重,按旧表里首次出现的顺序(稳定)
    new_names = []
    for tgt in resolved:
        if tgt != DELETE and tgt not in new_names:
            new_names.append(tgt)

    old2new = {}
    for i, tgt in enumerate(resolved):
        old2new[i] = DELETE if tgt == DELETE else new_names.index(tgt)
    return old, new_names, old2new


def rewrite_labels(dataset_dir, old2new, apply):
    changed = dropped = kept = 0
    for fp in iter_label_files(dataset_dir):
        out = []
        with open(fp, encoding="utf-8") as f:
            for ln in f:
                s = ln.strip()
                if not s:
                    continue
                parts = s.split()
                ci = int(parts[0])
                ni = old2new.get(ci, ci)
                if ni == DELETE:
                    dropped += 1
                    continue
                if ni != ci:
                    changed += 1
                parts[0] = str(ni)
                out.append(" ".join(parts))
                kept += 1
        if apply:
            with open(fp, "w", encoding="utf-8") as f:
                f.write("\n".join(out) + ("\n" if out else ""))
    return changed, dropped, kept


def write_outputs(dataset_dir, new_names, apply):
    if not apply:
        return
    with open(osp.join(dataset_dir, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(new_names) + "\n")
    # data.yaml(UTF-8;path 用正斜杠避免转义)
    lines = ["# 由 clean_review_classes.py 重建 — UTF-8",
             f"path: {dataset_dir}".replace("\\", "/"),
             "train: images/train", "val: images/val", "names:"]
    for i, n in enumerate(new_names):
        lines.append(f"  {i}: {n}")
    with open(osp.join(dataset_dir, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def backup(dataset_dir):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    bdir = osp.join(dataset_dir, f"_backup_clean_{stamp}")
    os.makedirs(bdir, exist_ok=True)
    for rel in ("classes.txt", "data.yaml"):
        src = osp.join(dataset_dir, rel)
        if osp.exists(src):
            shutil.copy(src, osp.join(bdir, rel))
    for _, lbl in IMG_LABEL_DIRS:
        src = osp.join(dataset_dir, lbl)
        if osp.isdir(src):
            shutil.copytree(src, osp.join(bdir, lbl.replace("/", "_")))
    return bdir


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--apply", action="store_true", help="真正写入(默认仅预演报告)")
    a = ap.parse_args(argv)

    old, new_names, old2new = plan(a.dataset)
    print(f"原类别数: {len(old)}  ->  清理后: {len(new_names)}")
    n_review = sum(1 for n in old if is_review_class(n))
    if n_review == 0:
        print("\n没有发现审查标记类(AI_REVIEW_* / missing_*),classes.txt 已是干净的。")
    else:
        print(f"\n[映射] {n_review} 个审查标记类的去向:")
        for i, n in enumerate(old):
            if is_review_class(n):
                ni = old2new[i]
                tgt = "删除该框" if ni == DELETE else f"-> {new_names[ni]} (新#{ni})"
                print(f"  旧#{i:2d} {n:55s} {tgt}")

    if a.apply:
        bdir = backup(a.dataset)
        print(f"\n已备份到: {bdir}")
    ch, dr, kp = rewrite_labels(a.dataset, old2new, a.apply)
    write_outputs(a.dataset, new_names, a.apply)
    verb = "已" if a.apply else "将"
    print(f"\n{verb}重写标注: 改类号 {ch} 个框, 删除 {dr} 个框, 保留 {kp} 个框")
    print(f"{verb}重建 classes.txt / data.yaml ({len(new_names)} 类, UTF-8)")
    if not a.apply:
        print("\n[预演] 没有写入任何文件。确认无误后加 --apply 执行。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
