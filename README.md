# SelfEvolvingRecognition

SelfEvolvingRecognition is a self-evolving recognition system built around
YOLO object detection. It is designed for camera streams, video streams, and
image datasets that need to improve over time without stopping recognition.

The system forms a closed loop:

1. Capture new images from recognition scenes.
2. Generate candidate annotations.
3. Let the user review annotation quality.
4. Publish only selected samples as a YOLO training dataset.
5. Train YOLO online while the current model keeps serving recognition.
6. Promote a better model, hold a worse model, or roll back manually.
7. Feed newly captured samples back into the next annotation round.

## Main Features

- Continuous data capture from cameras, videos, or image folders.
- Automatic annotation candidates for newly collected images.
- Manual review before data enters training.
- Quality triage with `keep`, `review`, and `reject` states.
- Selective dataset publishing: high-quality samples may be exported, skipped,
  or held for review by the user.
- Direct export to YOLO-ready `images`, `labels`, `classes.txt`, and
  `data.yaml` files.
- Online YOLO training while recognition remains active.
- Model registry with frozen validation metrics, promotion gates, and rollback.
- Hot-swap inference that loads a promoted model without interrupting frames.
- Detection-box smoothing during model updates and training convergence.

## System Loop

```text
camera / video / image dataset
        |
        v
capture useful frames
        |
        v
auto annotation + human review
        |
        v
quality triage: keep / review / reject
        |
        v
selective YOLO dataset publishing
        |
        v
online training + frozen validation
        |
        v
model registry: promote / hold / rollback
        |
        v
hot-swap inference without recognition interruption
        |
        v
new captured data enters the next loop
```

## Repository Layout

| Path | Purpose |
| --- | --- |
| `SelfEvolvingRecognition` | Main system implementation, GUI, annotation review, dataset publishing, online training, model registry, and live recognition. |
| `SelfEvolvingRecognition/anylabeling/services/auto_training` | Core self-evolving training and inference services. The package name is kept as an internal implementation detail. |
| `SelfEvolvingRecognition/anylabeling/views/training` | Training configuration and GUI integration. |
| `yolo` | YOLO sample data, base model, inference script, and reproduction notebooks. |

Important implementation files:

| File | Purpose |
| --- | --- |
| `collector.py` | Captures useful frames and prepares them for the next annotation round. |
| `publish_dataset.py` | Scans annotation quality, writes a manifest, and exports selected samples to YOLO format. |
| `train_with_registry.py` | Runs online training, evaluates on a frozen validation set, registers model versions, and supports rollback. |
| `model_registry.py` | Stores model versions, metrics, current model state, and rollback points. |
| `model_server.py` | Keeps inference alive while promoted models are loaded and hot-swapped. |
| `box_smoother.py` | Smooths detection boxes across frames and model switches. |
| `live_runtime.py` | Runs live recognition with model hot-swap and box smoothing. |

## Environment

Recommended setup:

- Windows 10/11
- Python 3.11 or 3.12
- CPU for full reproduction, GPU for faster training

```powershell
git clone https://github.com/Luoxr520/code.git
cd code

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip uv
```

Install the main system:

```powershell
cd SelfEvolvingRecognition
uv pip install -e ".[cpu]"
pip install -r ..\yolo\requirements.txt
ser checks
ser
cd ..
```

For GPU usage, replace `.[cpu]` with `.[gpu]` or `.[gpu-cu11]`. Use separate
virtual environments for CPU, CUDA 11, and CUDA 12 to avoid runtime conflicts.

## Full Reproduction

Run the commands below from the repository root unless noted otherwise.

### 1. Verify YOLO Inference

```powershell
cd yolo
python scripts\infer_coco128.py --model models\yolo26n.pt --source datasets\coco128\images\train2017 --output outputs\stage1_coco128_inference --device cpu --limit 20
cd ..
```

The output is written to `yolo/outputs/stage1_coco128_inference`.

### 2. Review Auto Annotations

```powershell
cd SelfEvolvingRecognition
ser ..\yolo\datasets\raw_images\unlabeled --output ..\yolo\datasets\labeled_yolo_v1 --labels ..\yolo\datasets\labeled_yolo_v1\classes.txt
cd ..
```

Use the GUI to review, correct, delete, or add boxes. Save the reviewed JSON
annotations before publishing a dataset.

### 3. Scan Annotation Quality

```powershell
cd SelfEvolvingRecognition
ser-publish scan ..\yolo\datasets\labeled_yolo_v1 --images-dir ..\yolo\datasets\raw_images\unlabeled --manifest ..\yolo\manifest.csv
cd ..
```

`manifest.csv` contains one row per image. The `publish` column controls
whether the sample enters the next training dataset.

Default triage:

- `keep`: selected by default for publishing.
- `review`: needs user review before publishing.
- `reject`: not selected for training.

The user can edit `publish` manually. A high-quality sample can still be held
back if it should not enter the next training round.

### 4. Publish a YOLO Dataset

```powershell
cd SelfEvolvingRecognition
ser-publish export ..\yolo\manifest.csv --out ..\yolo\datasets --images-dir ..\yolo\datasets\raw_images\unlabeled --per-class-val 1 --verify-images
cd ..
```

After publishing, `yolo/datasets` contains:

- `data.yaml`
- `classes.txt`
- `frozen_val.json`
- `images/train`
- `images/val`
- `labels/train`
- `labels/val`

### 5. Train Online With Registry Control

```powershell
cd SelfEvolvingRecognition
ser-train --dataset ..\yolo\datasets train --base ..\yolo\models\yolo26n.pt --epochs 10 --imgsz 640 --batch 8 --device cpu --workers 0 --promote-margin 0.0
cd ..
```

The training command evaluates the new model on the frozen validation split and
writes the decision to `yolo/datasets/registry/registry.json`.

List registered models:

```powershell
cd SelfEvolvingRecognition
ser-train --dataset ..\yolo\datasets list
cd ..
```

Roll back to a previous model:

```powershell
cd SelfEvolvingRecognition
ser-train --dataset ..\yolo\datasets rollback m_0001
cd ..
```

### 6. Run Live Recognition With Hot-Swap

Start recognition from a camera:

```powershell
cd SelfEvolvingRecognition
ser-live --dataset ..\yolo\datasets --source 0 --device cpu
cd ..
```

Run on a video file and save the result:

```powershell
cd SelfEvolvingRecognition
ser-live --dataset ..\yolo\datasets --source path\to\input.mp4 --save ..\yolo\outputs\live_result.mp4 --device cpu
cd ..
```

While live recognition is running, launch another training command. When a new
model passes the promotion gate, the live process loads and switches to it
without stopping recognition.

## Cleanup Policy

This repository keeps source code, reproduction scripts, required sample data,
and the base YOLO model. Runtime output should stay untracked:

- Python caches: `__pycache__`, `.pyc`
- IDE settings: `.vscode`, `.idea`
- Training outputs: `runs`, `weights/best.pt`, `weights/last.pt`
- Inference outputs: `outputs`
- Temporary registry, replay, and distillation artifacts
- Local camera captures and private video material
