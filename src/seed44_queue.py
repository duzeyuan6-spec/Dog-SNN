"""Cooperative GPU queue. Never kills/reconfigures other users' processes."""
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path
import psutil
import pynvml as nv
from common import config, write_json

STOP = False
ACTIVE = None


def stop_queue(*_):
    global STOP
    STOP = True
    if ACTIVE is not None and ACTIVE.poll() is None:
        try:
            os.killpg(ACTIVE.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


signal.signal(signal.SIGTERM, stop_queue)
signal.signal(signal.SIGINT, stop_queue)


def foreign_processes(handle, owned=()):
    found = {}
    for query in (nv.nvmlDeviceGetComputeRunningProcesses, nv.nvmlDeviceGetGraphicsRunningProcesses):
        for p in query(handle):
            if p.pid in owned:
                continue
            try:
                name = psutil.Process(p.pid).name()
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied:
                name = 'unknown'
            if name.lower() in ('xorg', 'xwayland'):
                continue
            found[p.pid] = name
    return found


def owned_pids(pid):
    try:
        return {pid} | {p.pid for p in psutil.Process(pid).children(recursive=True)}
    except psutil.NoSuchProcess:
        return {pid}


def main():
    global ACTIVE
    cfg = config()
    root = Path(cfg['root'])
    policy = cfg['resource_policy']
    lock = open(root/'seed44_queue.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    nv.nvmlInit()
    # Pilot v1 failed quality. Keep its artifacts, skip its expensive benchmark,
    # and advance to a separately versioned candidate without changing test gates.
    stages = [('seed44_'+s, ['seed_stage',s]) for s in ['v2','v3','v4','t2','t1']]
    stable = {}
    def state(status, **kwargs):
        write_json(root/'seed44_status.json', {'status': status, 'updated_at_unix': time.time(), **kwargs})
    while not (root/'prepared.json').exists():
        if STOP:
            state('stopped')
            return
        state('waiting_for_assets')
        time.sleep(policy['poll_seconds'])
    for name, args in stages:
        shared = args[0] == 'seed_stage' or args[0] == 'fast_quality' or args[0] in ('train', 'quality', 'iterate', 'audit', 'control', 'compare_quality', 'iterate_t2', 't2_quality', 'iterate_t1', 't1_quality')
        share_policy = dict(cfg['shared_training_policy'], gpu_indices=[6]) if args[0] in ('iterate_t1', 't1_quality') else cfg['shared_training_policy']
        if args[0]=='fast_quality': share_policy=dict(cfg['shared_training_policy'],gpu_indices=[2,3])
        share_policy = dict(cfg['shared_training_policy'],gpu_indices=[3])
        done = root/'runs'/f'{name}.done.json'
        if done.exists():
            continue
        while True:
            if STOP:
                state('stopped')
                return
            available = None
            observations = []
            for i in range(nv.nvmlDeviceGetCount()):
                handle = nv.nvmlDeviceGetHandleByIndex(i)
                # v1 includes driver-reserved VRAM; v2 separates actual allocations.
                memory_info = nv.nvmlDeviceGetMemoryInfo(handle, version=nv.nvmlMemory_v2)
                memory = memory_info.used/1024**2
                free = nv.nvmlDeviceGetMemoryInfo(handle).free/1024**2
                util = nv.nvmlDeviceGetUtilizationRates(handle).gpu
                others = foreign_processes(handle)
                idle = not others and memory < policy['max_memory_mb'] and util <= policy['max_utilization_percent']
                if shared:
                    idle = (i in share_policy['gpu_indices'] and free >= share_policy['min_free_memory_mb']
                            and util <= share_policy['max_utilization_percent'])
                stable[i] = stable.get(i, 0)+1 if idle else 0
                observations.append({'gpu': i, 'memory_mb': round(memory), 'utilization': util, 'driver_reserved_mb': round(memory_info.reserved/1024**2),
                                     'foreign_pids': list(others), 'idle_samples': stable[i]})
                if stable[i] >= policy['idle_samples'] and available is None:
                    available = (i, handle)
            if available is None:
                state('queued_waiting_for_gpu_headroom' if shared else 'queued_waiting_for_idle_gpu', stage=name, gpus=observations)
                time.sleep(policy['poll_seconds'])
                continue
            gpu, handle = available
            uuid = nv.nvmlDeviceGetUUID(handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode()
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=uuid, EXPERIMENT_NVML_UUID=uuid, OMP_NUM_THREADS='4',
                       EXPERIMENT_SHARED_GPU='1' if shared else '0')
            initial_compute = {p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(handle)}
            logpath = root/'logs'/f'{name}.log'
            with open(logpath, 'a', buffering=1) as log:
                script, script_args = ('audit_v4.py', []) if args[0] == 'audit' else (('iterate.py', []) if args[0] == 'iterate' else ('run.py', args))
                if args[0] == 'control':
                    script, script_args = 'ann_stage'+args[1]+'.py', []
                if args[0].startswith('compare_'):
                    script, script_args = 'evaluate_candidates.py', [args[0].split('_',1)[1]]
                if args[0] == 'compiled_performance':
                    script, script_args = 'evaluate_compiled.py', ['performance']
                if args[0] == 'iterate_t2':
                    script, script_args = 'iterate_t2.py', []
                if args[0].startswith('t2_'):
                    script, script_args = 'evaluate_t2.py', [args[0].split('_',1)[1]]
                if args[0] == 'iterate_t1':
                    script, script_args = 'iterate_t1.py', []
                if args[0].startswith('t1_'):
                    script, script_args = 'evaluate_t1.py', [args[0].split('_',1)[1]]
                if args[0] == 'repeat_t1':
                    script, script_args = 'evaluate_t1_repeat.py', ['performance']
                if args[0].startswith('fast_'):
                    script, script_args = 'evaluate_t1_fast.py', [args[0].split('_',1)[1]]
                if args[0]=='seed_stage':
                    script, script_args='train_seed.py',['44',args[1]]
                process = subprocess.Popen([str(root/'venv/bin/python'), '-u', str(root/'src'/script), *script_args],
                                           env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                ACTIVE = process
                state('running', stage=name, shared_gpu=shared, gpu=gpu, gpu_uuid=uuid, pid=process.pid, log=str(logpath))
                interrupted = False
                while process.poll() is None:
                    time.sleep(5)
                    others = foreign_processes(handle, owned_pids(process.pid))
                    should_yield = bool(others)
                    if shared:
                        free = nv.nvmlDeviceGetMemoryInfo(handle).free/1024**2
                        new_compute = {p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(handle)} - owned_pids(process.pid) - initial_compute
                        should_yield = free < share_policy['yield_below_free_mb'] or bool(new_compute)
                    if should_yield and process.poll() is None:
                        # Only signal the process group created immediately above.
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            process.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        interrupted = True
                        state('yielded_gpu_to_other_work', stage=name, gpu=gpu, foreign_pids=list(others))
                        break
                code = process.wait()
                ACTIVE = None
            stable = {}
            if interrupted or code == 75:
                time.sleep(policy['poll_seconds'])
                continue
            if code != 0:
                state('failed', stage=name, exit_code=code, log=str(logpath))
                raise SystemExit(code)
            write_json(done, {'finished_at_unix': time.time(), 'gpu_uuid': uuid})
            break
    state('candidate_complete_review_required', candidate='seed44/joint_t1_v6',
          report=str(root/'runs/seed44/joint_t1_v6/best_quality.json'), stage1_complete=False,
          next_action='Review validation and revise candidate if it fails; benchmark only qualified candidates')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write_json(Path(config()['root'])/'status.json', {'status': 'queue_failed',
                   'error': repr(error), 'updated_at_unix': time.time()})
        raise
