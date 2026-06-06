# -*- coding: utf-8 -*-
"""
model_server.py — 自我演化检测系统 · 第 3a 步(重设计)
双缓冲推理服务 + 注册表驱动的原子热替换 + 标准化推理接口。

这就是「训练时识别不中断」的发动机:
- get_model()/infer() 永远拿到一个已加载好、已 warmup 的稳定模型;
- 后台线程只在 registry.json 的 mtime 变化时重读;current 真的变了才换;
- 新模型在后台槽里加载 + warmup(锁外,耗时),就绪后只换指针(锁内,极快);
  → 加载新模型期间推理一直用旧模型,切换只是换指针,推理一刻不停;
- 新模型加载失败时保留旧模型,绝不让在线服务崩;
- warmup:新模型加载后先跑一帧假数据预热,避免热替换瞬间首帧卡顿。

单 8GB GPU 与训练共用显存时,建议 serving 放 CPU(device='cpu'):
训练独占 GPU,推理在 CPU 持续跑,换模型也在 CPU 侧,互不抢显存。

实时主循环(第 3c 步)用法骨架:
    server = ModelServer(dataset_dir, device='cpu')   # 与训练共用 GPU 时用 cpu
    server.load_current()
    server.start_watch(interval=2)        # 后台自动热替换
    while camera:
        dets = server.infer(frame)        # [Det,...] 原始检测框
        boxes = smoother.update(dets)     # 3b:连续过渡
        draw(frame, boxes); show()
    # 另开终端跑 train_with_registry train,新版本上线后这里自动热替换

ultralytics / numpy 懒加载:不加载真模型、不做推理的逻辑(轮询/换指针)无需它们即可测。
"""
from __future__ import annotations

import argparse
import os.path as osp
import sys
import threading
import time
from collections import namedtuple

try:
    from anylabeling.services.auto_training.model_registry import ModelRegistry
except ImportError:
    from model_registry import ModelRegistry

# 标准化的一个检测框:像素坐标 xyxy + 置信度 + 类别 id + 类别名
Det = namedtuple("Det", "x1 y1 x2 y2 conf cls label")

# 跟踪框:在 Det 基础上多一个稳定的 track id(tid)。ByteTrack 跨帧维持的同一目标 tid 不变。
# tid 可能为 None(ByteTrack 尚未确认该目标时),画框时按是否有 tid 区分显示。
TrackDet = namedtuple("TrackDet", "tid x1 y1 x2 y2 conf cls label")


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


def _arr(x):
    """把 torch.Tensor / numpy / list 统一成 numpy 数组,避免依赖 torch。"""
    import numpy as np

    if hasattr(x, "cpu"):     # torch.Tensor
        return x.cpu().numpy()
    return np.asarray(x)


def to_detections(result, names=None):
    """把 ultralytics 单帧结果转成 [Det,...]。names: {cls_id: 类别名}。"""
    names = names or {}
    out = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return out
    xyxy = _arr(boxes.xyxy)
    conf = _arr(boxes.conf)
    cls = _arr(boxes.cls)
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = (float(v) for v in xyxy[i][:4])
        k = int(cls[i])
        out.append(Det(x1, y1, x2, y2, float(conf[i]), k, names.get(k, str(k))))
    return out


def to_tracks(result, names=None):
    """把 ultralytics track() 单帧结果转成 [TrackDet,...]。
    读 boxes.id 作为稳定 track id;id 为 None(未确认/无跟踪)时该框 tid=None。"""
    names = names or {}
    out = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return out
    xyxy = _arr(boxes.xyxy)
    conf = _arr(boxes.conf)
    cls = _arr(boxes.cls)
    ids = getattr(boxes, "id", None)
    ids = _arr(ids) if ids is not None else None
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = (float(v) for v in xyxy[i][:4])
        k = int(cls[i])
        tid = int(ids[i]) if (ids is not None and i < len(ids)) else None
        out.append(TrackDet(tid, x1, y1, x2, y2, float(conf[i]), k, names.get(k, str(k))))
    return out


class ModelServer:
    def __init__(self, dataset_dir, device=None, loader=None,
                 conf=0.25, iou=0.45, imgsz=640, warmup=True, tracker="bytetrack.yaml"):
        self.dataset_dir = osp.abspath(dataset_dir)
        self.device = device
        self._loader = loader or self._default_loader
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.warmup = warmup
        self.tracker = tracker        # ByteTrack 配置(ultralytics 自带 bytetrack.yaml / botsort.yaml)

        self._lock = threading.Lock()
        self._active = None          # {"model","id","ckpt","metrics"}
        self._loaded_id = None
        self._reg_mtime = 0.0
        self._stop = threading.Event()
        self._thread = None

        self.on_swap = None          # 可选回调:on_swap(old_id, new_entry)
        self.swaps = 0
        self.last_swap_ts = None

    # ----------------------------------------------------------- 模型加载/预热
    def _default_loader(self, ckpt):
        from ultralytics import YOLO

        return YOLO(ckpt)

    def _warmup(self, model):
        """跑一帧假数据,预热 CUDA kernel;失败不影响服务。"""
        if not self.warmup:
            return
        try:
            import numpy as np

            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype="uint8")
            model.predict(dummy, device=self.device, imgsz=self.imgsz,
                          conf=self.conf, iou=self.iou, verbose=False)
        except Exception:
            pass

    def _load(self, ckpt):
        model = self._loader(ckpt)
        self._warmup(model)
        return model

    # ----------------------------------------------------------- 注册表读取
    def _reg_path(self):
        return osp.join(self.dataset_dir, "registry", "registry.json")

    def _current_if_changed(self):
        """只在 registry.json mtime 变化时重读;否则返回 None(跳过解析)。"""
        try:
            mt = osp.getmtime(self._reg_path())
        except OSError:
            return None
        if mt == self._reg_mtime and self._loaded_id is not None:
            return None
        self._reg_mtime = mt
        return ModelRegistry(self.dataset_dir).current()

    # ----------------------------------------------------------- 对外
    def load_current(self):
        """首帧前加载当前线上模型。"""
        cur = ModelRegistry(self.dataset_dir).current()
        if cur is None:
            raise RuntimeError("注册表里还没有任何模型;先用 train/eval 登记一个")
        try:
            self._reg_mtime = osp.getmtime(self._reg_path())
        except OSError:
            pass
        model = self._load(cur["ckpt"])
        with self._lock:
            self._active = {"model": model, "id": cur["id"],
                            "ckpt": cur["ckpt"], "metrics": cur.get("metrics", {})}
            self._loaded_id = cur["id"]
        return cur["id"]

    def load_ckpt(self, ckpt):
        """直接加载指定权重(不经注册表;用于测试/演示固定模型,不热替换)。"""
        model = self._load(ckpt)
        with self._lock:
            self._active = {"model": model, "id": osp.basename(ckpt),
                            "ckpt": ckpt, "metrics": {}}
            self._loaded_id = self._active["id"]
        return self._active["id"]

    def poll_and_swap(self):
        """current 变了则(锁外)加载+warmup,(锁内)原子换指针。返回新 id 或 None。"""
        cur = self._current_if_changed()
        if cur is None or cur["id"] == self._loaded_id:
            return None
        new_model = self._load(cur["ckpt"])   # 锁外:加载 + 预热(慢)
        old_id = self._loaded_id
        with self._lock:                      # 锁内:只换指针(快)
            self._active = {"model": new_model, "id": cur["id"],
                            "ckpt": cur["ckpt"], "metrics": cur.get("metrics", {})}
            self._loaded_id = cur["id"]
        self.swaps += 1
        self.last_swap_ts = time.time()
        if self.on_swap:
            try:
                self.on_swap(old_id, cur)
            except Exception:
                pass
        return cur["id"]

    def get_model(self):
        with self._lock:
            return self._active["model"] if self._active else None

    def infer(self, frame):
        """对一帧(numpy 图像或图片路径)推理,返回标准化检测列表 [Det,...]。"""
        model = self.get_model()
        if model is None:
            return []
        res = model.predict(frame, device=self.device, imgsz=self.imgsz,
                            conf=self.conf, iou=self.iou, verbose=False)
        r = res[0]
        names = getattr(model, "names", None) or getattr(r, "names", {}) or {}
        return to_detections(r, names)

    def track(self, frame):
        """对一帧做带 ByteTrack 的跟踪推理,返回带稳定 tid 的 [TrackDet,...]。

        必须按视频顺序逐帧调用(persist=True 跨帧维持轨迹);跳帧只会让 ByteTrack
        看到更低帧率的流,卡尔曼运动预测能扛。热替换换了模型对象后跟踪自然从头开始。
        注意:实时循环里只调 track()(不混用 predict),否则 ultralytics 可能重置跟踪器。
        """
        model = self.get_model()
        if model is None:
            return []
        res = model.track(frame, device=self.device, imgsz=self.imgsz,
                          conf=self.conf, iou=self.iou, persist=True,
                          tracker=self.tracker, verbose=False)
        r = res[0]
        names = getattr(model, "names", None) or getattr(r, "names", {}) or {}
        return to_tracks(r, names)

    def info(self):
        with self._lock:
            if not self._active:
                return None
            return {
                "id": self._active["id"],
                "ckpt": self._active["ckpt"],
                "metrics": dict(self._active["metrics"]),
                "swaps": self.swaps,
                "last_swap_ts": self.last_swap_ts,
            }

    # ----------------------------------------------------------- 后台轮询
    def start_watch(self, interval=2.0):
        def loop():
            while not self._stop.is_set():
                try:
                    self.poll_and_swap()
                except Exception:
                    pass   # 半写/瞬时错误:跳过,下个周期再试(不影响在线模型)
                self._stop.wait(interval)

        self._thread = threading.Thread(target=loop, name="model-hotswap", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


# --------------------------------------------------------------------------------------
# CLI:watch(演示热替换,无需摄像头)/ predict(用当前模型推理一张图)
# --------------------------------------------------------------------------------------
def cmd_watch(args):
    ds = resolve_dataset(args.dataset)
    if not ds:
        print("[serve] 没指定 --dataset,也没从 QSettings 读到发布输出目录", file=sys.stderr)
        return 1
    srv = ModelServer(ds, device=args.device, warmup=not args.no_warmup)

    def on_swap(old_id, entry):
        m = entry.get("metrics", {}).get("map5095")
        print(f"🔁 热替换: {old_id} -> {entry['id']}  (map5095={m})  {time.strftime('%H:%M:%S')}")

    srv.on_swap = on_swap
    srv.load_current()
    i = srv.info()
    print(f"[serve] 当前线上: {i['id']}  map5095={i['metrics'].get('map5095')}  device={args.device or 'auto'}")
    print(f"[serve] 后台每 {args.interval}s 检查注册表(mtime 变才读)。另开终端 train/rollback 即见热替换。Ctrl+C 退出。")
    srv.start_watch(interval=args.interval)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        srv.stop()
        print(f"\n[serve] 已停止(共热替换 {srv.swaps} 次)")
    return 0


def cmd_predict(args):
    ds = resolve_dataset(args.dataset)
    if not ds:
        print("[predict] 没指定 --dataset,也没从 QSettings 读到", file=sys.stderr)
        return 1
    srv = ModelServer(ds, device=args.device, conf=args.conf, iou=args.iou)
    srv.load_current()
    dets = srv.infer(args.image)
    print(f"[predict] 模型 {srv.info()['id']} 在 {args.image} 检出 {len(dets)} 个框:")
    for d in dets[:50]:
        print(f"   {d.label:>12}  conf={d.conf:.2f}  [{d.x1:.0f},{d.y1:.0f},{d.x2:.0f},{d.y2:.0f}]")
    return 0


def build_parser():
    p = argparse.ArgumentParser(description="双缓冲推理服务 + 注册表驱动热替换")
    sub = p.add_subparsers(dest="cmd", required=True)

    # 公共参数挂到各子命令上,这样 --dataset/--device 跟在子命令后面也能用
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", default=None, help="数据集目录(默认读 QSettings 的发布输出目录)")
    common.add_argument("--device", default=None, help="0 / cpu;与训练共用 GPU 时建议 cpu")

    w = sub.add_parser("watch", parents=[common], help="演示热替换:加载当前模型并后台监听注册表变化")
    w.add_argument("--interval", type=float, default=2.0, help="轮询注册表的间隔秒数")
    w.add_argument("--no-warmup", action="store_true", help="关闭加载后预热")
    w.set_defaults(func=cmd_watch)

    pr = sub.add_parser("predict", parents=[common], help="用当前线上模型对一张图做一次推理")
    pr.add_argument("image")
    pr.add_argument("--conf", type=float, default=0.25)
    pr.add_argument("--iou", type=float, default=0.45)
    pr.set_defaults(func=cmd_predict)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
