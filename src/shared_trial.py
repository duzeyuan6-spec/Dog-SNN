"""User-authorized shared-card diagnostics. Never eligible for final acceptance."""
import os,json,time,signal,threading,hashlib
from pathlib import Path
import numpy as np
import torch
import pynvml as nv
from common import config,seed_all,teacher,Frames,write_json
from spike_encoder import SpikeEncoder
from resource_queue import foreign_processes
STOP=False
INITIAL=set()
def stop(*_):
 global STOP
 STOP=True
signal.signal(signal.SIGTERM,stop)
def guard(handle):
 if STOP: raise SystemExit(75)
 current=set(foreign_processes(handle,{os.getpid()}))
 if current-INITIAL or nv.nvmlDeviceGetMemoryInfo(handle,version=nv.nvmlMemory_v2).free < 8*1024**3:
  raise SystemExit(75)

def benchmark(fn, x, handle):
    guard(handle)
    for _ in range(20):
        fn(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    latencies = []
    wall_latencies = []
    for _ in range(150):
        if STOP:
            raise SystemExit(75)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter()
        begin.record()
        fn(x)
        end.record()
        torch.cuda.synchronize()
        wall_latencies.append((time.perf_counter()-wall)*1000)
        latencies.append(begin.elapsed_time(end))
    rounds = []
    for _ in range(3):
        guard(handle)
        samples = []
        sampling_stop = threading.Event()
        def power_sample():
            while not sampling_stop.is_set():
                try:
                    samples.append((time.perf_counter(), nv.nvmlDeviceGetPowerUsage(handle)/1000))
                except nv.NVMLError:
                    pass
                sampling_stop.wait(0.05)
        thread = threading.Thread(target=power_sample, daemon=True)
        thread.start()
        try:
            energy_start = nv.nvmlDeviceGetTotalEnergyConsumption(handle)
        except nv.NVMLError:
            energy_start = None
        t0 = time.perf_counter()
        count = 0
        while time.perf_counter()-t0 < 10:
            if STOP:
                sampling_stop.set()
                thread.join()
                raise SystemExit(75)
            for _ in range(16):
                fn(x)
            torch.cuda.synchronize()
            count += 16
        elapsed = time.perf_counter()-t0
        try:
            energy_end = nv.nvmlDeviceGetTotalEnergyConsumption(handle) if energy_start is not None else None
        except nv.NVMLError:
            energy_end = None
        sampling_stop.set()
        thread.join()
        guard(handle)
        method, energy = 'unavailable', None
        if energy_end is not None and energy_end > energy_start:
            method, energy = 'nvml_total_energy_mj', (energy_end-energy_start)/1000
        elif len(samples) >= 2:
            # Include endpoints by constant extension; retain raw samples for audit.
            ts, ps = zip(*samples)
            inside = [(max(t0, min(t0+elapsed, t)), p) for t, p in zip(ts, ps)]
            ts = [t0]+[t for t, _ in inside]+[t0+elapsed]
            ps = [ps[0]]+[p for _, p in inside]+[ps[-1]]
            energy = float(np.trapezoid(ps, ts))
            method = 'nvml_board_power_integral_50ms_polling'
        rounds.append({'images': count, 'seconds': elapsed, 'images_per_second': count/elapsed,
                       'gpu_joules_per_image': energy/count if energy is not None else None,
                       'energy_method': method, 'power_samples': samples})
    return {'scope': 'GPU-resident model forward; excludes file IO, CPU preprocessing and transfers',
            'precision': 'float32, TF32 disabled', 'batch_size': 1,
            'cuda_p50_ms': float(np.median(latencies)), 'cuda_p95_ms': float(np.percentile(latencies, 95)),
            'wall_p50_ms': float(np.median(wall_latencies)), 'wall_p95_ms': float(np.percentile(wall_latencies, 95)),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'rounds': rounds}


@torch.inference_mode()
def main():
 global INITIAL
 cfg=config();seed_all(42);root=Path(cfg['root']);nv.nvmlInit()
 h=nv.nvmlDeviceGetHandleByUUID(os.environ['EXPERIMENT_NVML_UUID'])
 INITIAL=set(foreign_processes(h,{os.getpid()}));guard(h)
 out=root/'runs/t1_shared_diagnostic'/os.environ['TRIAL_GPU']/str(time.time_ns());out.mkdir(parents=True)
 write_json(out/'metadata.json',dict(shared_gpu=True,acceptance_eligible=False,gpu_uuid=os.environ['EXPERIMENT_NVML_UUID'],initial_foreign_pids=sorted(INITIAL),energy_scope='Whole GPU board including other processes; not isolated model energy'))
 done=threading.Event()
 def monitor():
  with (out/'occupancy.jsonl').open('a',buffering=1) as f:
   while not done.is_set():
    m=nv.nvmlDeviceGetMemoryInfo(h,version=nv.nvmlMemory_v2)
    f.write(json.dumps(dict(time=time.time(),foreign_pids=sorted(foreign_processes(h,{os.getpid()})),used=m.used,free=m.free,util=nv.nvmlDeviceGetUtilizationRates(h).gpu,power_w=nv.nvmlDeviceGetPowerUsage(h)/1000))+'\n');done.wait(1)
 t=threading.Thread(target=monitor,daemon=True);t.start()
 try:
  torch.cuda.set_per_process_memory_fraction(.25)
  ref=teacher(cfg,'cuda');original={k:v.detach().cpu().clone() for k,v in ref.decoder.state_dict().items()}
  path=root/'runs/joint_t1_v6/best.pt';sha=hashlib.sha256(path.read_bytes()).hexdigest()
  prior=json.loads((root/'runs/t1_comparison/quality_summary.json').read_text());assert sha==prior['checkpoint_hashes']['snn']
  ck=torch.load(path,map_location='cpu',weights_only=False);sp=ck['spec'];model=SpikeEncoder(sp['width'],sp['depth'],sp['timesteps']).cuda().eval();model.load_state_dict(ck['model']);model.set_blend(1)
  x=Frames(cfg,'validation')[0][0].unsqueeze(0).cuda();blocks=[]
  for block,order in enumerate([['baseline','snn'],['snn','baseline']]*3):
   measurements={}
   for kind in order:
    guard(h);ref.decoder.load_state_dict(original if kind=='baseline' else ck['decoder'])
    def fn(x):
     z=ref.encode(x).latent_dist.mode() if kind=='baseline' else model(x)[0]
     return ref.decode(z).sample
    measurements[kind]=benchmark(fn,x,h)
    print(json.dumps(dict(block=block,kind=kind,p95_ms=measurements[kind]['wall_p95_ms'],shared_gpu=True)),flush=True)
   blocks.append(dict(order=order,measurements=measurements));write_json(out/f'block_{block}.json',blocks[-1])
  write_json(out/'summary.json',dict(blocks=blocks,checkpoint_sha256=sha,shared_gpu=True,acceptance_eligible=False,stage1_complete=False))
  print(json.dumps(dict(event='shared_trial_complete',output=str(out))),flush=True)
 finally:done.set();t.join(timeout=3)
if __name__=='__main__':
 import fcntl
 root=Path(config()['root']);label=os.environ['TRIAL_GPU']
 lock=(root/('shared_trial_'+label+'.lock')).open('a')
 try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError: raise SystemExit(0)
 done=root/'runs'/('shared_trial_'+label+'.done.json')
 if not done.exists():
  main()
  write_json(done,dict(completed_at=time.time(),shared_gpu=True,acceptance_eligible=False))

