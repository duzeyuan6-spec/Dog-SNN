"""Detached server supervisor: finish environment, verify, prepare, then queue."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    cfg = json.loads(Path(__file__).with_name('config.json').read_text())
    root = Path(cfg['root'])
    python = root/'venv/bin/python'
    env = os.environ.copy()
    env.update(UV_CACHE_DIR=str(root/'uv-cache'), UV_PYTHON_INSTALL_DIR=str(root/'python'),
               TORCH_HOME=str(root/'models/torch-cache'), HF_HOME=str(root/'hf-cache'),
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    def status(value, **kwargs):
        tmp = root/'status.json.tmp'
        tmp.write_text(json.dumps({'status': value, 'updated_at_unix': time.time(), **kwargs}, indent=2))
        os.replace(tmp, root/'status.json')
    def ready():
        return python.exists() and subprocess.run([str(python), '-c',
            'import torch,torchvision,av,pyarrow,lpips,pynvml,psutil; from diffusers import AutoencoderKLFlux2'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env).returncode == 0
    original_pid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    while not ready():
        status('queued_waiting_for_python_environment')
        try:
            cmdline = Path(f'/proc/{original_pid}/cmdline').read_bytes()
            original_running = str(root).encode() in cmdline and b'uv' in cmdline
        except FileNotFoundError:
            original_running = False
        if original_running:
            time.sleep(30)
        else:
            with open(root/'logs/install.log', 'a') as log:
                subprocess.run([str(root/'tools/bin/uv'), 'pip', 'install', '--python', str(python),
                                '-r', str(root/'src/requirements.txt')], env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            if not ready():
                raise RuntimeError('Environment installed but required imports failed')
    status('running_cpu_verification')
    with open(root/'logs/smoke.log', 'a') as log:
        subprocess.run([str(python), '-u', str(root/'src/smoke.py')], env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    with open(root/'environment.json', 'w') as out:
        subprocess.run([str(python), '-c',
            'import importlib.metadata as m,json; print(json.dumps({d.metadata["Name"]:d.version for d in m.distributions()},indent=2))'],
            env=env, stdout=out, check=True)
    while not (root/'assets_downloaded.json').exists():
        status('queued_waiting_for_model_and_data_downloads')
        time.sleep(30)
    if not (root/'prepared.json').exists():
        status('preparing_dataset_on_cpu')
        with open(root/'logs/prepare.log', 'a') as log:
            subprocess.run([str(python), '-u', str(root/'src/prepare.py')], env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    # Exec retains the supervisor PID and keeps the queue under this tmux session.
    os.execve(str(python), [str(python), '-u', str(root/'src/resource_queue.py')], env)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        cfg = json.loads(Path(__file__).with_name('config.json').read_text())
        (Path(cfg['root'])/'status.json').write_text(json.dumps({'status': 'setup_failed',
            'error': repr(error), 'updated_at_unix': time.time()}, indent=2))
        raise
