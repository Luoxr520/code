# -*- coding: utf-8 -*-
"""
train_with_registry.py — 自我演化检测系统 · 第 2 步 runner
训练 -> 在冻结验证集上评测 -> 进注册表过 promote/回退闸门。

子命令:
  train     从「当前 current」的权重继续训(冷启动则用 --base),训完在冻结 val 上评测,提交注册表过闸门
  eval      对一个已存在的 .pt 在冻结 val 上评测并提交注册表(给「用现成训练界面训出来的模型」补登记用)
  list      打印注册表(各版本指标/状态,★ = 当前线上)
  rollback  把 current 回退到指定版本(由你决定)
  current   打印当前线上模型权重路径(给推理/第 3 步 serving 用)

数据集路径:--dataset 指定;不指定则自动读 SelfEvolvingRecognition 持久化的 QSettings(publish/output_dir),
也就是你在发布面板里设的那个输出目录。数据集里自带 data.yaml/classes.txt/frozen_val.json。

ultralytics 为懒加载:不训练的子命令(list/rollback/current)无需装 ultralytics 即可用。
"""
from __future__ import annotations

import argparse
import os.path as osp
import sys
import time
from pathlib import Path

try:  # 既能在 SelfEvolvingRecognition 包内,也能在 evo_system 文件夹里直接跑
    from anylabeling.services.auto_training.model_registry import (
        ModelRegistry,
        PRIMARY_METRIC,
    )
except ImportError:
    from model_registry import ModelRegistry, PRIMARY_METRIC


# --------------------------------------------------------------------------------------
# 数据集定位 + 工具
# --------------------------------------------------------------------------------------
def resolve_dataset(arg):
    """显式 --dataset 优先;否则读 SelfEvolvingRecognition 持久化的发布输出目录。"""
    if arg:
        return osp.abspath(arg)
    try:
        from PyQt6.QtCore import QSettings

        v = QSettings("anylabeling", "anylabeling").value("publish/output_dir", "")
        if v:
            return osp.abspath(str(v))
    except Exception:
        pass
    return None


def _data_yaml(dataset_dir):
    p = osp.join(dataset_dir, "data.yaml")
    if not osp.exists(p):
        raise FileNotFoundError(
            f"在 {dataset_dir} 找不到 data.yaml。先用发布面板/CLI 导出数据集。"
        )
    return p


def _read_classes(dataset_dir):
    p = osp.join(dataset_dir, "classes.txt")
    if osp.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    return []


def _metrics_from_val(res):
    """从 ultralytics 的 val 结果对象提取检测指标(纯函数,可单测)。"""
    box = res.box
    return {
        "map5095": float(box.map),
        "map50": float(box.map50),
        "map75": float(getattr(box, "map75", float("nan"))),
        "precision": float(box.mp),
        "recall": float(box.mr),
    }


# --------------------------------------------------------------------------------------
# ultralytics 调用(懒加载,真训练/评测)
# --------------------------------------------------------------------------------------
def evaluate_ckpt(ckpt, data_yaml, device=None, imgsz=640, workers=0):
    """在冻结 val(data.yaml 里的 val:)上评测一个 checkpoint,返回指标 dict。
    workers=0:数据加载在主进程,避免 Windows 上 DataLoader 子进程加载 CUDA DLL
    触发『页面文件太小』(WinError 1455)。数据集大时可在 CLI 用 --workers 调高。"""
    from ultralytics import YOLO

    model = YOLO(ckpt)
    res = model.val(
        data=data_yaml, split="val", device=device, imgsz=imgsz,
        workers=workers, verbose=False,
    )
    return _metrics_from_val(res)


def _epoch_metrics(metrics):
    """从 ultralytics trainer.metrics 稳健取 (map50, map5095)。"""
    m = metrics or {}
    map5095 = next((v for k, v in m.items() if "mAP50-95" in k), None)
    map50 = next((v for k, v in m.items() if "mAP50" in k and "mAP50-95" not in k), None)
    return map50, map5095


def _epoch_line(epoch, metrics):
    """组装给 GUI 解析的每轮精度行:'@EPOCH <ep> <map50> <map5095>'(纯函数,便于测试)。"""
    map50, map5095 = _epoch_metrics(metrics)

    def fmt(v):
        return f"{float(v):.5f}" if isinstance(v, (int, float)) else "nan"

    return f"@EPOCH {int(epoch)} {fmt(map50)} {fmt(map5095)}"


def _compose_note(dataset_dir, data_yaml, epochs, replay, stats, freeze, lr0):
    """自动备注:训练数据组成(用户/回放张数)+ 轮数 + 抗遗忘杠杆 + 回放源。"""
    parts = []
    try:
        from importlib import import_module
        try:
            R = import_module("anylabeling.services.auto_training.replay")
        except ImportError:
            import replay as R
        _, _, train_dir, _ = R._resolve_yaml(data_yaml)
        n_user = len(R.yolo_image_label_pairs(train_dir))
        n_replay = stats["replay"] if (replay and stats) else 0
        parts.append(f"用户{n_user}张" + (f"+回放{n_replay}张" if replay else ""))
    except Exception:
        pass
    parts.append(f"ep{epochs}")
    if freeze is not None:
        parts.append(f"freeze{freeze}")
    if lr0 is not None:
        parts.append(f"lr{lr0}")
    if replay:
        srcs = ",".join(osp.basename(str(s).rstrip("/\\")) for s in replay)
        if srcs:
            parts.append("源:" + srcs)
    return " · ".join(parts)


def train_once(dataset_dir, base=None, epochs=50, imgsz=640, batch=16,
               device=None, note="", promote_margin=0.0, workers=0,
               replay=None, replay_max=0, freeze=None, lr0=None, optimizer=None):
    """从当前 current 继续训(没有则用 base),训完在冻结 val 上评测并提交注册表过闸门。
    replay: 回放数据源目录列表(混入训练集抗遗忘);训练用合成集,评测仍用冻结验证集。
    freeze/lr0: 抗遗忘杠杆(冻结主干层数 / 续训用小学习率)。
    optimizer: SGD/AdamW/auto 等;注意 auto 会忽略 lr0,故指定了 lr0 而没给 optimizer 时自动用 SGD。"""
    from ultralytics import YOLO

    data_yaml = _data_yaml(dataset_dir)            # 冻结验证集来自这里,评测始终用它
    reg = ModelRegistry(dataset_dir, promote_margin=promote_margin)

    start_from = reg.current_ckpt() or base or "yolo11n.pt"
    parent_id = reg.current_id  # 记录这次是从哪个版本继续训的
    print(f"[train] 数据集: {dataset_dir}")
    print(f"[train] 起始权重: {start_from}  (current_id={parent_id})")

    # 回放:把外部/历史样本(按类名重映射)混进训练集;val 仍是冻结验证集
    train_data_yaml = data_yaml
    stats = None
    if replay:
        from importlib import import_module
        try:
            R = import_module("anylabeling.services.auto_training.replay")
        except ImportError:
            import replay as R  # 独立运行兜底
        work_dir = osp.join(dataset_dir, ".replay_work")
        train_data_yaml, stats = R.compose_train_dir(
            dataset_dir, data_yaml, list(replay), replay_max, work_dir
        )
        print(f"[train] 回放: 训练集 = 用户 {stats['user']} 张 + 回放 {stats['replay']} 张  (源: {', '.join(replay)})")

    run_name = f"train_{time.strftime('%Y%m%d_%H%M%S')}"
    model = YOLO(start_from)

    # 每轮验证结束发一行可解析的精度数据,供 GUI 实时画折线图(终端里也是一行,无害)
    def _on_fit_epoch_end(trainer):
        try:
            ep = int(getattr(trainer, "epoch", 0)) + 1
            print(_epoch_line(ep, getattr(trainer, "metrics", None)), flush=True)
        except Exception:
            pass

    try:
        model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
    except Exception:
        pass

    train_kwargs = dict(
        data=train_data_yaml,
        epochs=epochs, imgsz=imgsz, batch=batch, device=device, workers=workers,
        project=osp.join(dataset_dir, "runs"), name=run_name, exist_ok=True,
    )
    if freeze is not None:
        train_kwargs["freeze"] = freeze
        print(f"[train] 冻结前 {freeze} 层(抗遗忘:保留预训练主干)")
    if lr0 is not None:
        train_kwargs["lr0"] = lr0
        print(f"[train] lr0={lr0}(续训小学习率,减少漂移)")
    # optimizer 解析:ultralytics 的 optimizer=auto 会忽略 lr0。
    # 若用户给了 lr0 但没指定 optimizer,则自动切到 SGD,否则 lr0 形同虚设。
    eff_optimizer = optimizer
    if eff_optimizer is None and lr0 is not None:
        eff_optimizer = "SGD"
        print("[train] 检测到设置了 lr0 但未指定优化器;自动用 SGD(auto 会忽略 lr0)")
    if eff_optimizer is not None:
        train_kwargs["optimizer"] = eff_optimizer
        print(f"[train] 优化器: {eff_optimizer}")
    results = model.train(**train_kwargs)

    save_dir = Path(getattr(results, "save_dir", "") or model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"训练没产出 best.pt:{best}")

    print(f"[train] 训练完成,在冻结验证集上评测 {best.name} ...")
    metrics = evaluate_ckpt(str(best), data_yaml, device=device, imgsz=imgsz, workers=workers)

    # 自动备注:把训练数据组成等信息写进备注(用户可在 GUI 里再编辑)
    auto_note = _compose_note(dataset_dir, data_yaml, epochs, replay, stats, freeze, lr0)
    final_note = (note + " | " + auto_note) if note else auto_note
    print(f"[train] 备注: {final_note}")

    entry, decision = reg.submit(
        str(best), metrics,
        parent_id=parent_id, classes=_read_classes(dataset_dir),
        epochs=epochs, base=start_from, note=final_note,
    )
    return reg, entry, decision


def find_latest_run(dataset_dir):
    """返回 <dataset>/runs/train_* 里最新的训练目录;没有则 None。"""
    import glob as _glob

    runs = [r for r in _glob.glob(osp.join(dataset_dir, "runs", "train_*")) if osp.isdir(r)]
    return max(runs, key=osp.getmtime) if runs else None


def _run_is_registered(reg, run_dir):
    """该 run 的 best.pt 是否已登记进注册表(按 src_ckpt 判断,避免重复登记)。"""
    run_dir = osp.abspath(run_dir)
    for e in reg.list():
        src = e.get("src_ckpt") or ""
        if src and osp.abspath(src).startswith(run_dir):
            return True
    return False


def register_run(dataset_dir, run_dir, device=None, imgsz=640, workers=0,
                 promote_margin=0.0, note="resume 登记"):
    """评测某次训练的 best.pt 并登记(不重训)。返回 (reg, entry, decision);已登记则 entry=None。"""
    best = osp.join(run_dir, "weights", "best.pt")
    if not osp.exists(best):
        return None
    reg = ModelRegistry(dataset_dir, promote_margin=promote_margin)
    if _run_is_registered(reg, run_dir):
        print(f"[resume] {osp.basename(run_dir)} 的 best.pt 已在注册表里,跳过")
        return reg, None, None
    print(f"[resume] 训练已完成,直接评测 + 登记(不重训): {best}")
    metrics = evaluate_ckpt(best, _data_yaml(dataset_dir), device=device, imgsz=imgsz, workers=workers)
    entry, decision = reg.submit(
        best, metrics, classes=_read_classes(dataset_dir), base=best, note=note
    )
    return reg, entry, decision


def _print_decision(reg, entry, decision):
    m = decision["metric"]
    cv = decision["candidate"]
    print(f"\n[结果] 新版本 {entry['id']}  {m}={cv}")
    if decision["promoted"]:
        if decision["reason"] == "improved":
            print(f"[闸门] 优于当前线上模型(Δ{decision['delta']:+.4f})-> 已上线 ✅  current={decision['current_id']}")
        else:
            print(f"[闸门] {decision['reason']} -> 已上线 ✅  current={decision['current_id']}")
    else:
        if decision["reason"] == "no_improvement":
            print(f"[闸门] 不优于当前线上模型(Δ{decision['delta']:+.4f})-> 未上线 ⛔,current 仍为 {decision['current_id']}")
            print(f"       该版本已留存为可回退历史。要强制启用它:rollback {entry['id']}")
        else:
            print(f"[闸门] {decision['reason']} -> 未上线,current 仍为 {decision['current_id']}")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def cmd_train(args):
    ds = resolve_dataset(args.dataset)
    if not ds:
        print("[train] 没指定 --dataset,也没从 QSettings 读到发布输出目录", file=sys.stderr)
        return 1
    reg, entry, decision = train_once(
        ds, base=args.base, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, note=args.note, promote_margin=args.promote_margin,
        workers=args.workers, replay=args.replay, replay_max=args.replay_max,
        freeze=args.freeze, lr0=args.lr0, optimizer=args.optimizer,
    )
    _print_decision(reg, entry, decision)
    return 0


def cmd_seed_replay(args):
    ds = resolve_dataset(args.dataset)
    if not ds:
        print("[seed-replay] 没指定 --dataset,也没从 QSettings 读到", file=sys.stderr)
        return 1
    from importlib import import_module
    try:
        R = import_module("anylabeling.services.auto_training.replay")
    except ImportError:
        import replay as R
    total = 0
    for src in args.from_:
        n = R.seed_into_buffer(src, ds, cap=args.cap, seed=args.seed)
        print(f"[seed-replay] {src}: 加入 {n} 张 -> {R.buffer_dir(ds)}")
        total += n
    print(f"[seed-replay] 合计 {total} 张")
    return 0


def cmd_distill(args):
    ds = resolve_dataset(args.dataset)
    if not ds:
        print("[distill] 没指定 --dataset,也没从 QSettings 读到", file=sys.stderr)
        return 1
    reg = ModelRegistry(ds)
    t = args.teacher
    if t in (None, "current"):
        teacher_ckpt = reg.current_ckpt()
        if not teacher_ckpt:
            print("[distill] 注册表里还没有当前模型,无法当老师", file=sys.stderr)
            return 1
    elif str(t).startswith("m_"):
        teacher_ckpt = next((e["ckpt"] for e in reg.list() if e["id"] == t), None)
        if not teacher_ckpt:
            print(f"[distill] 找不到版本 {t}", file=sys.stderr)
            return 1
    else:
        teacher_ckpt = t  # 当作权重路径
    out = args.out or osp.join(ds, "distill")
    from importlib import import_module
    try:
        R = import_module("anylabeling.services.auto_training.replay")
    except ImportError:
        import replay as R
    n_img, n_box = R.generate_pseudo_labels(
        ds, teacher_ckpt, args.images, out,
        conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device,
    )
    print(f"[distill] 老师 {osp.basename(teacher_ckpt)} 在 {args.images} 生成 {n_box} 个伪标注框,覆盖 {n_img} 张图 -> {out}")
    print(f"[distill] 训练时加  --replay \"{out}\"  即可把老师的知识蒸馏进去")
    return 0


def cmd_eval(args):
    ds = resolve_dataset(args.dataset)
    if not ds:
        print("[eval] 没指定 --dataset,也没从 QSettings 读到发布输出目录", file=sys.stderr)
        return 1
    reg = ModelRegistry(ds, promote_margin=args.promote_margin)
    metrics = evaluate_ckpt(args.ckpt, _data_yaml(ds), device=args.device, imgsz=args.imgsz, workers=args.workers)
    entry, decision = reg.submit(
        args.ckpt, metrics, classes=_read_classes(ds), base=args.ckpt, note=args.note or "外部模型"
    )
    _print_decision(reg, entry, decision)
    return 0


def cmd_resume(args):
    ds = resolve_dataset(args.dataset)
    if not ds:
        print("[resume] 没指定 --dataset,也没从 QSettings 读到发布输出目录", file=sys.stderr)
        return 1
    run_dir = args.run or find_latest_run(ds)
    if not run_dir or not osp.isdir(run_dir):
        print(f"[resume] 在 {osp.join(ds, 'runs')} 找不到任何训练目录,无可恢复", file=sys.stderr)
        return 1
    print(f"[resume] 目标训练目录: {run_dir}")
    best = osp.join(run_dir, "weights", "best.pt")
    last = osp.join(run_dir, "weights", "last.pt")

    # 显式续训:训练中途被打断,从 last.pt 继续到设定 epoch,再评测+登记
    if args.continue_training and osp.exists(last):
        from ultralytics import YOLO

        print("[resume] 从 last.pt 续训到设定 epoch ...")
        model = YOLO(last)
        results = model.train(resume=True)
        save_dir = Path(getattr(results, "save_dir", "") or model.trainer.save_dir)
        bp = save_dir / "weights" / "best.pt"
        reg = ModelRegistry(ds, promote_margin=args.promote_margin)
        metrics = evaluate_ckpt(str(bp), _data_yaml(ds), device=args.device, imgsz=args.imgsz, workers=args.workers)
        entry, decision = reg.submit(str(bp), metrics, classes=_read_classes(ds), base=str(bp), note="resume 续训")
        _print_decision(reg, entry, decision)
        return 0

    # 默认:训练已完成(有 best.pt)-> 只补评测+登记,绝不重训(你要的那个场景)
    if osp.exists(best):
        out = register_run(ds, run_dir, device=args.device, imgsz=args.imgsz,
                           workers=args.workers, promote_margin=args.promote_margin)
        if out and out[1] is not None:
            _print_decision(*out)
        return 0

    if osp.exists(last):
        print("[resume] 只找到 last.pt(训练很早就中断了)。加 --continue-training 续训完成。", file=sys.stderr)
        return 1

    print("[resume] 该目录里没有 best.pt / last.pt,无可恢复。", file=sys.stderr)
    return 1


def cmd_list(args):
    ds = resolve_dataset(args.dataset)
    if not ds:
        print("[list] 没指定 --dataset,也没从 QSettings 读到发布输出目录", file=sys.stderr)
        return 1
    reg = ModelRegistry(ds)
    rows = reg.summary()
    if not rows:
        print(f"[list] 注册表为空:{osp.join(ds, 'registry')}")
        return 0
    m = reg.metric
    print(f"数据集: {ds}   主指标: {m}")
    print(f"{'cur':<4}{'id':<9}{m:<10}{'status':<12}{'parent':<9}{'time':<20}note")
    for r in rows:
        v = r[m]
        v = f"{v:.4f}" if isinstance(v, (int, float)) else "-"
        print(f"{r['current']:<4}{r['id']:<9}{v:<10}{r['status']:<12}{str(r['parent']):<9}{r['ts']:<20}{r['note']}")
    return 0


def cmd_rollback(args):
    ds = resolve_dataset(args.dataset)
    if not ds:
        print("[rollback] 没指定 --dataset,也没从 QSettings 读到", file=sys.stderr)
        return 1
    reg = ModelRegistry(ds)
    reg.rollback(args.id)
    print(f"[rollback] current 已设为 {args.id};权重: {reg.current_ckpt()}")
    return 0


def cmd_reeval_all(args):
    """在【当前】冻结验证集上,把注册表里每个版本的权重重新评测、更新分数,
    然后重新决定 current = 新验证集下主指标最高者。换了验证集(如注入数据)后用这个让分数可比。"""
    ds = resolve_dataset(args.dataset)
    if not ds:
        print("[reeval-all] 没指定 --dataset,也没从 QSettings 读到", file=sys.stderr)
        return 1
    reg = ModelRegistry(ds)
    data_yaml = _data_yaml(ds)
    rows = reg.list()
    if not rows:
        print("[reeval-all] 注册表为空")
        return 0
    old_cur = reg.current_id
    m = reg.metric
    print(f"[reeval-all] 在当前验证集上重评 {len(rows)} 个版本 ...  (数据集: {ds})")
    for e in rows:
        ckpt = e["ckpt"]
        if not ckpt or not osp.exists(ckpt):
            print(f"  {e['id']}: 权重缺失,跳过 ({ckpt})", file=sys.stderr)
            continue
        old_v = e["metrics"].get(m)
        metrics = evaluate_ckpt(ckpt, data_yaml, device=args.device,
                                imgsz=args.imgsz, workers=args.workers)
        reg.update_metrics(e["id"], metrics)
        ov = f"{old_v:.4f}" if isinstance(old_v, (int, float)) else "-"
        nv = metrics.get(m)
        nv = f"{nv:.4f}" if isinstance(nv, (int, float)) else "-"
        print(f"  {e['id']}: {m} {ov} -> {nv}")
    new_cur = reg.recompute_current()
    print(f"[reeval-all] 重评完成。current: {old_cur} -> {new_cur}")
    if new_cur != old_cur:
        print(f"[reeval-all] 注意:新验证集下最优版本变了,current 已更新为 {new_cur}")
    return 0


def cmd_current(args):
    ds = resolve_dataset(args.dataset)
    if not ds:
        print("[current] 没指定 --dataset,也没从 QSettings 读到", file=sys.stderr)
        return 1
    reg = ModelRegistry(ds)
    cur = reg.current()
    if not cur:
        print("[current] 注册表还没有任何模型")
        return 1
    print(cur["ckpt"])
    return 0


def build_parser():
    p = argparse.ArgumentParser(description="训练 -> 冻结验证集评测 -> 注册表 promote/回退")
    p.add_argument("--dataset", default=None, help="数据集目录(默认读 QSettings 里发布面板设的输出目录)")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="从当前 current 继续训,评测后过闸门")
    t.add_argument("--base", default="yolo11n.pt", help="冷启动起始权重(可指向本地 yolo26n.pt 等)")
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--device", default=None, help="0 / cpu;默认自动")
    t.add_argument("--workers", type=int, default=0, help="DataLoader 进程数;Windows 页面文件小就保持 0")
    t.add_argument("--promote-margin", type=float, default=0.0, help="候选需超过当前多少才上线")
    t.add_argument("--replay", action="append", default=None, help="回放数据源目录(可多次);如 coco128 路径,或 <dataset>\\replay")
    t.add_argument("--replay-max", type=int, default=0, help="每轮最多混入多少张回放样本(0=全部)")
    t.add_argument("--freeze", type=int, default=None, help="冻结前 N 层(抗遗忘:保留预训练主干,如 10)")
    t.add_argument("--lr0", type=float, default=None, help="初始学习率(续训用小值减少漂移,如 0.001)")
    t.add_argument("--optimizer", default=None,
                   help="SGD/AdamW/auto 等;注意 auto 会忽略 lr0。给了 lr0 而不指定时自动用 SGD")
    t.add_argument("--note", default="")
    t.set_defaults(func=cmd_train)

    sr = sub.add_parser("seed-replay", help="把外部数据集(如 coco128)按类名重映射存进回放缓冲 <dataset>/replay")
    sr.add_argument("--from", dest="from_", action="append", required=True, help="源数据集目录(可多次)")
    sr.add_argument("--cap", type=int, default=None, help="每个源最多取多少张")
    sr.add_argument("--seed", type=int, default=0)
    sr.set_defaults(func=cmd_seed_replay)

    dt = sub.add_parser("distill", help="老师模型给(无标注)图片打伪标注,供训练蒸馏(LwF)")
    dt.add_argument("--teacher", default="current", help="current / 版本号(如 m_0007)/ 权重路径")
    dt.add_argument("--images", required=True, help="要打伪标注的图片目录(如 raw_images\\unlabeled)")
    dt.add_argument("--out", default=None, help="输出目录(默认 <dataset>/distill)")
    dt.add_argument("--conf", type=float, default=0.4, help="保留该置信度以上预测当伪标注")
    dt.add_argument("--iou", type=float, default=0.45)
    dt.add_argument("--imgsz", type=int, default=640)
    dt.add_argument("--device", default=None)
    dt.set_defaults(func=cmd_distill)

    e = sub.add_parser("eval", help="评测一个已有 .pt 并提交注册表")
    e.add_argument("ckpt")
    e.add_argument("--imgsz", type=int, default=640)
    e.add_argument("--device", default=None)
    e.add_argument("--workers", type=int, default=0, help="DataLoader 进程数;Windows 页面文件小就保持 0")
    e.add_argument("--promote-margin", type=float, default=0.0)
    e.add_argument("--note", default="")
    e.set_defaults(func=cmd_eval)

    rs = sub.add_parser("resume", help="恢复被中断的运行:训练已完成则只补评测+登记(不重训)")
    rs.add_argument("--run", default=None, help="指定训练目录;默认取 runs/ 下最新的一个")
    rs.add_argument("--continue-training", action="store_true", help="从 last.pt 续训到设定 epoch(训练中途被打断时用)")
    rs.add_argument("--imgsz", type=int, default=640)
    rs.add_argument("--device", default=None)
    rs.add_argument("--workers", type=int, default=0)
    rs.add_argument("--promote-margin", type=float, default=0.0)
    rs.set_defaults(func=cmd_resume)

    li = sub.add_parser("list", help="打印注册表")
    li.set_defaults(func=cmd_list)

    re_ = sub.add_parser("reeval-all", help="在当前验证集上重评所有版本并重判 current(换验证集后用)")
    re_.add_argument("--device", default=None)
    re_.add_argument("--imgsz", type=int, default=640)
    re_.add_argument("--workers", type=int, default=0)
    re_.set_defaults(func=cmd_reeval_all)

    rb = sub.add_parser("rollback", help="把 current 回退到指定版本")
    rb.add_argument("id")
    rb.set_defaults(func=cmd_rollback)

    cu = sub.add_parser("current", help="打印当前线上模型权重路径")
    cu.set_defaults(func=cmd_current)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
