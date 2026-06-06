# -*- coding: utf-8 -*-
"""
live_runtime.py — 自我演化检测系统 · 第 3c 步
实时主循环:摄像头/视频 → 当前线上模型推理(后台自动热替换)→ 检测框连续过渡 → 画框+角标 → 显示/写出。

把 3a(ModelServer 热替换)+ 3b(BoxSmoother 连续过渡)串成完整效果:
一边后台训练持续产出更好的模型自动热替换,一边画面里框越收越准、识别不中断。

用法:
    # 摄像头实时显示(与训练共用 GPU 时 serving 建议 cpu)
    python -m anylabeling.services.auto_training.live_runtime --source 0 --device cpu
    # 跑视频文件并写出标注视频
    python -m anylabeling.services.auto_training.live_runtime --source in.mp4 --save out.mp4
另开一个终端持续训练,画面里会自动热替换:
    python -m anylabeling.services.auto_training.train_with_registry train --epochs 50 --device 0 --workers 0

角标显示:模型版本 / mAP50-95 / 热替换次数 / FPS / 当前帧检测数 / 跟踪数,换模型瞬间顶部高亮提示。
注:cv2.putText 只画 ASCII,类别名是中文时会显示乱码(可后续用 PIL 画中文)。

数据流核心(step / hud_lines / FpsMeter 等)不依赖 cv2,便于测试;cv2 读帧/画框/显示在本机跑。
"""
from __future__ import annotations

import argparse
import os.path as osp
import sys
import time

try:
    from anylabeling.services.auto_training.model_server import ModelServer, resolve_dataset
    from anylabeling.services.auto_training.box_smoother import BoxSmoother
except ImportError:
    from model_server import ModelServer, resolve_dataset
    from box_smoother import BoxSmoother

# 类别配色(BGR),按 cls 取
PALETTE = [
    (66, 135, 245), (80, 200, 120), (60, 76, 231), (235, 180, 52),
    (180, 120, 255), (255, 200, 80), (52, 235, 210), (200, 80, 180),
    (120, 200, 60), (90, 90, 235),
]


def color_for_class(cls):
    return PALETTE[int(cls) % len(PALETTE)]


def clamp_box(x1, y1, x2, y2, w, h):
    """裁到画面内并转成 int。"""
    xi1 = max(0, min(int(round(x1)), w - 1))
    yi1 = max(0, min(int(round(y1)), h - 1))
    xi2 = max(0, min(int(round(x2)), w - 1))
    yi2 = max(0, min(int(round(y2)), h - 1))
    if xi2 < xi1:
        xi1, xi2 = xi2, xi1
    if yi2 < yi1:
        yi1, yi2 = yi2, yi1
    return xi1, yi1, xi2, yi2


class FpsMeter:
    """EMA 平滑的 FPS 计。tick() 内部用 perf_counter;测试可传 now。"""

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.last = None
        self.fps = 0.0

    def tick(self, now=None):
        now = now if now is not None else time.perf_counter()
        if self.last is not None:
            dt = now - self.last
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if self.fps == 0.0 else self.alpha * inst + (1 - self.alpha) * self.fps
        self.last = now
        return self.fps


def hud_lines(info, fps, ndet, ntracks):
    """角标文字(ASCII)。info 来自 server.info()。"""
    if info:
        mid = info.get("id", "?")
        mp = (info.get("metrics", {}) or {}).get("map5095")
        swaps = info.get("swaps", 0)
    else:
        mid, mp, swaps = "?", None, 0
    mp_s = f"{mp:.3f}" if isinstance(mp, (int, float)) else "-"
    return [
        f"Model {mid}   mAP50-95 {mp_s}   swaps {swaps}",
        f"FPS {fps:.1f}   dets {ndet}   tracks {ntracks}",
    ]


def step(server, smoother, frame, fpsm, state, now=None):
    """每帧的核心处理(无 cv2):推理 → 平滑 → 角标。返回 (dets, sboxes, lines, banner)。"""
    dets = server.infer(frame)
    sboxes = smoother.update(dets)
    t = now if now is not None else time.time()
    banner = None
    b = state.get("banner")
    if b and t < b[1]:
        banner = b[0]
    lines = hud_lines(server.info(), fpsm.tick(now), len(dets), len(sboxes))
    return dets, sboxes, lines, banner


# --------------------------------------------------------------------------------------
# cv2 绘制(本机运行)
# --------------------------------------------------------------------------------------
def _draw_one(img, x1, y1, x2, y2, color, label):
    import cv2

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    by2 = y1
    by1 = max(0, y1 - th - bl - 4)
    cv2.rectangle(img, (x1, by1), (x1 + tw + 4, by2), color, -1)
    cv2.putText(img, label, (x1 + 2, by2 - bl - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_boxes(img, sboxes):
    import cv2

    h, w = img.shape[:2]
    for b in sboxes:
        color = color_for_class(b.cls)
        x1, y1, x2, y2 = clamp_box(b.x1, b.y1, b.x2, b.y2, w, h)
        label = f"{b.id} {b.label} {b.conf:.2f}"
        if b.alpha >= 0.99:
            _draw_one(img, x1, y1, x2, y2, color, label)
        else:                       # 渐隐渐显:按 alpha 混合这一个框
            ov = img.copy()
            _draw_one(ov, x1, y1, x2, y2, color, label)
            cv2.addWeighted(ov, float(b.alpha), img, 1 - float(b.alpha), 0, img)


def draw_hud(img, lines, banner=None):
    import cv2

    pad, fs, lh = 8, 0.6, 24
    texts = ([banner] if banner else []) + list(lines)
    tw = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0] for t in texts)
    pw, ph = tw + 2 * pad, lh * len(texts) + pad
    ov = img.copy()
    cv2.rectangle(ov, (8, 8), (8 + pw, 8 + ph), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
    y = 8 + pad + 14
    for i, t in enumerate(texts):
        col = (0, 215, 255) if (banner and i == 0) else (235, 235, 235)
        cv2.putText(img, t, (8 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, fs, col, 1, cv2.LINE_AA)
        y += lh


# --------------------------------------------------------------------------------------
# 主循环
# --------------------------------------------------------------------------------------
def run(args):
    try:
        import cv2
    except ImportError:
        print("需要 opencv-python:pip install opencv-python", file=sys.stderr)
        return 1

    ds = resolve_dataset(args.dataset)
    if not ds:
        print("[live] 没指定 --dataset,也没从 QSettings 读到发布输出目录", file=sys.stderr)
        return 1

    server = ModelServer(ds, device=args.device, conf=args.conf, iou=args.iou, imgsz=args.imgsz)
    state = {"banner": None}

    def on_swap(old_id, entry):
        m = (entry.get("metrics", {}) or {}).get("map5095")
        m_s = f"{m:.3f}" if isinstance(m, (int, float)) else "-"
        state["banner"] = (f"swapped {old_id} -> {entry['id']}  mAP {m_s}", time.time() + 2.5)
        print(f"🔁 {state['banner'][0]}")

    server.on_swap = on_swap
    try:
        sid = server.load_current()
    except Exception as ex:  # noqa: BLE001
        print("[live] 加载当前模型失败:", ex, file=sys.stderr)
        return 1
    print(f"[live] 当前线上: {sid}  device={args.device or 'auto'}  source={args.source}")
    server.start_watch(interval=args.interval)

    smoother = BoxSmoother(smooth=args.smooth, iou_match=args.iou_match,
                           fade_in=args.fade_in, fade_out=args.fade_out, max_age=args.max_age)
    fpsm = FpsMeter()

    src = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("[live] 无法打开视频源:", args.source, file=sys.stderr)
        server.stop()
        return 1

    win = "self-evolving detection"
    if not args.no_window:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    writer = None
    frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            _, sboxes, lines, banner = step(server, smoother, frame, fpsm, state)
            draw_boxes(frame, sboxes)
            draw_hud(frame, lines, banner)

            if args.save:
                if writer is None:
                    h, w = frame.shape[:2]
                    fps_out = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    if fps_out <= 1:
                        fps_out = 25.0
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps_out, (w, h))
                writer.write(frame)
            if not args.no_window:
                cv2.imshow(win, frame)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
            frames += 1
            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_window:
            cv2.destroyAllWindows()
        server.stop()
    sw = server.info()["swaps"] if server.info() else 0
    print(f"[live] 结束,共 {frames} 帧,热替换 {sw} 次" + (f",已写出 {args.save}" if args.save else ""))
    return 0


def build_parser():
    p = argparse.ArgumentParser(description="实时主循环:推理(热替换)+ 检测框连续过渡")
    p.add_argument("--source", default="0", help="摄像头编号(0)或视频文件路径")
    p.add_argument("--dataset", default=None, help="数据集目录(默认读 QSettings)")
    p.add_argument("--device", default=None, help="0 / cpu;与训练共用 GPU 时建议 cpu")
    p.add_argument("--interval", type=float, default=2.0, help="热替换轮询间隔秒")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--smooth", type=float, default=0.4, help="EMA 系数:大跟手小平滑")
    p.add_argument("--iou-match", type=float, default=0.3, help="跟踪关联 IoU 阈值")
    p.add_argument("--fade-in", type=int, default=3)
    p.add_argument("--fade-out", type=int, default=6)
    p.add_argument("--max-age", type=int, default=8, help="目标丢失多少帧后移除")
    p.add_argument("--save", default=None, help="写出标注视频路径(.mp4)")
    p.add_argument("--no-window", action="store_true", help="不显示窗口(配合 --save)")
    p.add_argument("--max-frames", type=int, default=0, help="跑这么多帧后停(0=不限)")
    return p


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
