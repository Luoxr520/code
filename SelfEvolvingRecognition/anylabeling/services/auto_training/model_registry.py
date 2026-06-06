# -*- coding: utf-8 -*-
"""
model_registry.py — 自我演化检测系统 · 第 2 步核心
模型注册表 + 基于冻结验证集的 promote / 回退闸门。

放在数据集目录下的 registry/ 里(数据集由第 1 步发布,自带 data.yaml/classes.txt/frozen_val.json):
    <dataset>/registry/registry.json   每个 checkpoint 的指标、血缘、状态;current_id 指向当前线上模型
    <dataset>/registry/models/<id>.pt  每个登记模型的副本(自包含、可移植,runs 被清也不丢)

判定逻辑(全部在本文件,可单测,不依赖 torch/ultralytics):
- 第一个模型直接上线;
- 之后的候选,只有在冻结验证集主指标(默认 mAP50-95)上「不劣于当前线上模型 + 不小于 promote_margin」才上线;
- 否则保留为可回退的历史版本,current 不变(防止「训练后反而更差」被推上线);
- rollback(to_id):由用户决定,把任意历史版本设回 current。
"""
from __future__ import annotations

import json
import os
import os.path as osp
import shutil
import time

# 检测任务的主指标候选;默认用 mAP50-95
PRIMARY_METRIC = "map5095"


class ModelRegistry:
    def __init__(self, dataset_dir, metric=PRIMARY_METRIC, promote_margin=0.0):
        self.dataset_dir = osp.abspath(dataset_dir)
        self.reg_dir = osp.join(self.dataset_dir, "registry")
        self.models_dir = osp.join(self.reg_dir, "models")
        self.path = osp.join(self.reg_dir, "registry.json")
        os.makedirs(self.models_dir, exist_ok=True)
        self._data = self._load(metric, promote_margin)

    # ----------------------------------------------------------- 持久化
    def _load(self, metric, promote_margin):
        if osp.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "dataset": self.dataset_dir,
            "metric": metric,
            "promote_margin": float(promote_margin),
            "current_id": None,
            "entries": [],
        }

    def save(self):
        # 原子落盘:先写临时文件再 os.replace,避免另一进程(推理服务)读到半截 JSON
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    # ----------------------------------------------------------- 只读属性
    @property
    def metric(self):
        return self._data["metric"]

    @property
    def promote_margin(self):
        return float(self._data.get("promote_margin", 0.0))

    @property
    def current_id(self):
        return self._data.get("current_id")

    def _next_id(self):
        existing = {e["id"] for e in self._data["entries"]}
        n = len(existing) + 1
        while f"m_{n:04d}" in existing:
            n += 1
        return f"m_{n:04d}"

    def _find(self, eid):
        for e in self._data["entries"]:
            if e["id"] == eid:
                return e
        return None

    def current(self):
        return self._find(self.current_id) if self.current_id else None

    def current_ckpt(self):
        """当前线上模型的权重路径:训练下一轮从它继续(持续学习),或作为 serving 模型。"""
        e = self.current()
        return e["ckpt"] if e else None

    # ----------------------------------------------------------- 登记
    def _store_ckpt(self, ckpt, eid):
        """把 checkpoint 复制进 registry/models,使注册表自包含、可移植。"""
        if not ckpt or not osp.exists(ckpt):
            return osp.abspath(ckpt) if ckpt else None
        dst = osp.join(self.models_dir, f"{eid}.pt")
        try:
            shutil.copy2(ckpt, dst)
            return dst
        except Exception:
            return osp.abspath(ckpt)

    def add(self, ckpt, metrics, status="candidate", parent_id="__current__",
            classes=None, n_train=None, n_val=None, epochs=None, base=None, note=""):
        eid = self._next_id()
        stored = self._store_ckpt(ckpt, eid)
        entry = {
            "id": eid,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ckpt": stored,
            "src_ckpt": osp.abspath(ckpt) if ckpt else None,
            "parent_id": self.current_id if parent_id == "__current__" else parent_id,
            "metrics": metrics or {},
            "classes": classes or [],
            "n_train": n_train,
            "n_val": n_val,
            "epochs": epochs,
            "base": base,
            "note": note,
            "status": status,
        }
        self._data["entries"].append(entry)
        self.save()
        return entry

    def register_seed(self, ckpt, metrics=None, note="seed"):
        """登记冷启动基础模型(如 YOLO 蒸出的初值或 COCO 预训练)为初始 current。"""
        entry = self.add(ckpt, metrics or {}, status="seed", parent_id=None, note=note)
        self._data["current_id"] = entry["id"]
        self.save()
        return entry

    # ----------------------------------------------------------- 闸门
    def propose(self, entry_id):
        """用冻结验证集主指标与当前 current 比较,决定 promote(上线)还是 hold(保留可回退)。"""
        cand = self._find(entry_id)
        if cand is None:
            raise ValueError(f"未知模型 id: {entry_id}")
        m = self.metric
        cand_v = cand["metrics"].get(m)
        cur = self.current()

        if cur is None:
            cand["status"] = "promoted"
            self._data["current_id"] = cand["id"]
            self.save()
            return self._decision(True, "first_model", cand_v, None, None)

        cur_v = cur["metrics"].get(m)
        if cand_v is None:
            cand["status"] = "candidate"
            self.save()
            return self._decision(False, "candidate_metric_missing", None, cur_v, None)
        if cur_v is None:
            cand["status"] = "promoted"
            self._data["current_id"] = cand["id"]
            self.save()
            return self._decision(True, "current_metric_missing", cand_v, None, None)

        delta = cand_v - cur_v
        # 严格更优才上线:delta 必须 > promote_margin(同分不替换,避免无意义的 current 漂移)
        if delta > self.promote_margin:
            cand["status"] = "promoted"
            cur["status"] = "superseded"
            self._data["current_id"] = cand["id"]
            self.save()
            return self._decision(True, "improved", cand_v, cur_v, delta)
        else:
            # 训练后没有严格变好 -> 不上线,保留为可回退版本
            cand["status"] = "rejected"
            self.save()
            return self._decision(False, "no_improvement", cand_v, cur_v, delta)

    def _decision(self, promoted, reason, cand_v, cur_v, delta):
        return {
            "promoted": promoted,
            "reason": reason,
            "metric": self.metric,
            "candidate": cand_v,
            "current": cur_v,
            "delta": delta,
            "current_id": self.current_id,
        }

    def submit(self, ckpt, metrics, **kw):
        """登记候选 + 立即过闸门。返回 (entry, decision)。训练 runner 用这个。"""
        e = self.add(ckpt, metrics, status="candidate", **kw)
        d = self.propose(e["id"])
        return e, d

    def rollback(self, to_id):
        """用户决定:把任意历史版本设回 current。"""
        if self._find(to_id) is None:
            raise ValueError(f"未知模型 id: {to_id}")
        for e in self._data["entries"]:
            if e["id"] == to_id:
                e["status"] = "promoted"
            elif e.get("status") == "promoted":
                e["status"] = "superseded"
        self._data["current_id"] = to_id
        self.save()
        return True

    def set_note(self, entry_id, note):
        """改某个版本的备注并原子落盘。"""
        e = self._find(entry_id)
        if e is None:
            raise ValueError(f"未知模型 id: {entry_id}")
        e["note"] = note
        self.save()
        return True

    def update_metrics(self, entry_id, metrics):
        """更新某版本的指标(如换了验证集后重评);不改状态,不落盘(由调用方批量后统一 save)。"""
        e = self._find(entry_id)
        if e is None:
            raise ValueError(f"未知模型 id: {entry_id}")
        e["metrics"] = dict(metrics or {})
        return e

    def recompute_current(self):
        """在所有版本【当前指标】下重新决定 current = 主指标最高者。
        换验证集重评后调用:让 current 反映新验证集下的真实最优。返回新的 current id。
        重评会重置旧裁决:所有【有指标】的版本都参与竞争(包括曾被 rejected 的——
        旧的 rejected 可能只是因为旧验证集算得低)。当选者 promoted,其余有分的 superseded。"""
        m = self.metric
        scored = [(e, e["metrics"].get(m)) for e in self._data["entries"]
                  if e["metrics"].get(m) is not None]
        if not scored:
            self.save()
            return self.current_id
        # 选主指标最高;并列时取更早登记的(id 最小,稳定)
        best = max(scored, key=lambda t: (t[1], -int(t[0]["id"].split("_")[-1])))[0]
        for e, _v in scored:
            e["status"] = "promoted" if e is best else "superseded"
        self._data["current_id"] = best["id"]
        self.save()
        return best["id"]

    # ----------------------------------------------------------- 查询
    def list(self):
        cur = self.current_id
        rows = []
        for e in self._data["entries"]:
            ee = dict(e)
            ee["is_current"] = e["id"] == cur
            rows.append(ee)
        return rows

    def summary(self):
        """给 CLI / GUI 展示用的精简行。"""
        m = self.metric
        out = []
        for e in self.list():
            out.append({
                "id": e["id"],
                "ts": e["ts"],
                m: e["metrics"].get(m),
                "status": e["status"],
                "current": "★" if e["is_current"] else "",
                "parent": e.get("parent_id"),
                "note": e.get("note", ""),
            })
        return out
