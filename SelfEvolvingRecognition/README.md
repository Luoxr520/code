# SelfEvolvingRecognition

This directory contains the main implementation of the self-evolving YOLO
recognition system.

The application combines annotation review, selective dataset publishing,
online training, model registry control, rollback, and live recognition
hot-swap. It is intended to keep recognition running while new data is reviewed
and used to improve the model.

## Install

```powershell
cd SelfEvolvingRecognition
py -3.12 -m venv .venv-cpu
.\.venv-cpu\Scripts\Activate.ps1
python -m pip install -U pip uv
uv pip install -e ".[cpu]"
pip install -r ..\yolo\requirements.txt
ser checks
ser
```

For GPU environments, install `.[gpu]` or `.[gpu-cu11]` instead of `.[cpu]`.

## Commands

| Command | Purpose |
| --- | --- |
| `ser` | Launch the GUI or open an image directory. |
| `ser-publish` | Scan annotation quality and export selected samples to YOLO format. |
| `ser-train` | Train, evaluate, register, promote, list, and roll back models. |
| `ser-live` | Run live recognition with model hot-swap and box smoothing. |

## Core Files

| File | Purpose |
| --- | --- |
| `anylabeling/services/auto_training/publish_dataset.py` | Quality scan and selective YOLO dataset export. |
| `anylabeling/services/auto_training/train_with_registry.py` | Online training and registry decisions. |
| `anylabeling/services/auto_training/model_registry.py` | Versioned model registry with rollback. |
| `anylabeling/services/auto_training/model_server.py` | Hot-swap inference service. |
| `anylabeling/services/auto_training/live_runtime.py` | Live recognition loop. |
| `anylabeling/services/auto_training/box_smoother.py` | Stable detection-box transitions. |
| `anylabeling/services/auto_training/collector.py` | Capture helpers for the next data loop. |

## Reproduction

The full reproduction flow is documented in the repository root `README.md`.
