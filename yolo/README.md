# YOLO Reproduction

This directory contains the YOLO sample dataset, base model, inference script,
and notebooks used to reproduce the recognition loop.

## Layout

| Path | Purpose |
| --- | --- |
| `datasets/coco128` | Small sample dataset for inference checks. |
| `datasets/raw_images/unlabeled` | Images prepared for annotation review. |
| `datasets/labeled_yolo_v1` | Reviewed annotations and class file. |
| `models` | Local YOLO model files. |
| `scripts/infer_coco128.py` | Direct inference script. |
| `scripts/stage1_yolo_coco128_inference.ipynb` | Stage 1 notebook for YOLO inference. |
| `scripts/stage2_auto_labeling_system.ipynb` | Stage 2 notebook for annotation review. |

## Install

```powershell
cd yolo
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

Install the main system before running annotation review:

```powershell
cd ..\SelfEvolvingRecognition
pip install -e ".[cpu]"
ser checks
cd ..\yolo
```

Use `[gpu]` or `[gpu-cu11]` instead of `[cpu]` when your environment supports it.

## Run Inference

```powershell
python scripts\infer_coco128.py --device cpu
```

Common full command:

```powershell
python scripts\infer_coco128.py --model models\yolo26n.pt --source datasets\coco128\images\train2017 --output outputs\stage1_coco128_inference --conf 0.25 --limit 20
```

Outputs:

- `outputs/stage1_coco128_inference/annotated_images`
- `outputs/stage1_coco128_inference/json_results`
- `outputs/stage1_coco128_inference/all_detections.json`

## Start Annotation Review

```powershell
ser datasets\raw_images\unlabeled --output datasets\labeled_yolo_v1 --labels datasets\labeled_yolo_v1\classes.txt
```

Review and save annotations in the GUI, then return to the root README flow to
scan quality, publish the YOLO dataset, train, and run live recognition.
