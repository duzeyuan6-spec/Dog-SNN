#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# ponytail: reuse the host's working PyTorch/OpenCV; Jetson CUDA wheels are platform-specific.
"${FOLLOWER_PYTHON:-/usr/bin/python3}" -m venv --system-site-packages --without-pip .venv
requirements=requirements.txt
if [[ "$(uname -m)" == "aarch64" ]]; then
  requirements=requirements-jetson.txt
fi
.venv/bin/python -m pip install --no-cache-dir --no-deps --only-binary=:all: -r "$requirements"
.venv/bin/python -c 'import torch, cv2, pyorbbecsdk, ultralytics, lap, polars; print("Dependencies ready")'
