#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# ponytail: reuse the robot's PyTorch and pyorbbecsdk; do not download another CUDA stack.
python3 -m venv --system-site-packages --without-pip .venv
.venv/bin/python -m pip install --no-cache-dir --no-deps -r requirements.txt
.venv/bin/python -c 'import torch, cv2, pyorbbecsdk, ultralytics, lap, polars; print("Dependencies ready")'
