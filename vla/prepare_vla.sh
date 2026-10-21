#!/usr/bin/env bash
set -euo pipefail
ROOT=/datasets2/dzy/snn_vae_20261015
VLA=$ROOT/vla
export UV_CACHE_DIR=$VLA/uv-cache UV_PYTHON_INSTALL_DIR=$VLA/python HF_HOME=$VLA/hf-cache
UV=$ROOT/tools/bin/uv
if [ ! -x "$VLA/venv/bin/python" ]; then "$UV" venv --python 3.10 "$VLA/venv"; fi
"$UV" pip install --python "$VLA/venv/bin/python" 'numpy<2' setuptools wheel packaging ninja
TF_REV=bc339d9ad707454c0c115970db43c260067c61ab
if [ ! -f "$VLA/vendor/transformers-openvla-oft/setup.py" ]; then
  curl -fL --retry 5 --retry-delay 3 "https://codeload.github.com/moojink/transformers-openvla-oft/tar.gz/$TF_REV" -o "$VLA/vendor/transformers-oft.tar.gz"
  mkdir -p "$VLA/vendor/transformers-openvla-oft"
  tar -xzf "$VLA/vendor/transformers-oft.tar.gz" --strip-components=1 -C "$VLA/vendor/transformers-openvla-oft"
fi
printf 'transformers @ file://%s/vendor/transformers-openvla-oft\n' "$VLA" > "$VLA/overrides.txt"
"$UV" pip install --override "$VLA/overrides.txt" --python "$VLA/venv/bin/python" -e "$VLA/vendor/openvla-oft" 'numpy<2'
while [ ! -f "$VLA/vendor/LIBERO/setup.py" ]; do sleep 10; done
"$UV" pip install --python "$VLA/venv/bin/python" -e "$VLA/vendor/LIBERO" -r "$VLA/vendor/openvla-oft/experiments/robot/libero/libero_requirements.txt"
"$UV" pip install --python "$VLA/venv/bin/python" tensorflow-metadata==1.15.0 wandb==0.16.6 mujoco==2.3.7
"$UV" pip freeze --python "$VLA/venv/bin/python" > "$VLA/environment.lock.txt"
