# Local Setup

This directory is the source tree for SelfEvolvingRecognition. Python 3.11 or
newer is required; Python 3.12 is recommended.

## Windows CPU Environment

```powershell
cd SelfEvolvingRecognition
py -3.12 -m venv .venv-cpu
.\.venv-cpu\Scripts\Activate.ps1
python -m pip install -U pip uv
uv pip install -e ".[cpu]"
```

Verify the installation:

```powershell
ser checks
ser version
```

Launch the GUI:

```powershell
ser
```

## Windows GPU Environment

CUDA 12:

```powershell
cd SelfEvolvingRecognition
py -3.12 -m venv .venv-cu12
.\.venv-cu12\Scripts\Activate.ps1
python -m pip install -U pip uv
uv pip install -e ".[gpu]"
ser
```

CUDA 11:

```powershell
cd SelfEvolvingRecognition
py -3.12 -m venv .venv-cu11
.\.venv-cu11\Scripts\Activate.ps1
python -m pip install -U pip uv
uv pip install -e ".[gpu-cu11]"
ser
```

Keep CPU, CUDA 11, and CUDA 12 installs in separate virtual environments.

## Development

```powershell
uv pip install -e ".[cpu,dev]"
ser help
ser checks
ser config
pytest
```

## Troubleshooting

- `ser` is not found: activate the virtual environment and reinstall with
  `uv pip install -e ".[cpu]"`.
- GUI startup fails: create a fresh virtual environment and reinstall.
- GPU is unavailable: verify the driver and runtime package, then fall back to
  the CPU environment if needed.
- Moved source directory: run the editable install command again.
