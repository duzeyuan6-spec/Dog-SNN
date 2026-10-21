"""Matched observation replay: full host-image to processed action-chunk cost."""
import os,sys,time,json,pickle,subprocess,hashlib,atexit
from pathlib import Path
import numpy as np
R=Path('/datasets2/dzy/snn_vae_20261015');V=R/'vla'
sys.path[:0]=[str(V/'vendor/openvla-oft'),str(V/'vendor/LIBERO')]
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import torch,pynvml as nv,psutil
from experiments.robot.libero import run_libero_eval as e
out=Path(os.environ['PIPELINE_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
shared=os.environ.get('EXPERIMENT_SHARED_GPU')=='1'
nv.nvmlInit();h=nv.nvmlDeviceGetHandleByIndex(int(os.environ['CUDA_VISIBLE_DEVICES']))
def guard():
 owned={os.getpid()}|{p.pid for p in psutil.Process().children(recursive=True)}
 foreign=[p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(h) if p.pid not in owned]
 m=nv.nvmlDeviceGetMemoryInfo(h,version=nv.nvmlMemory_v2)
 if m.free < 8000*1024**2 or (foreign and not shared):raise RuntimeError('resource guard: insufficient free memory or foreign process on exclusive GPU')
 return {'foreign_pids':foreign,'free_bytes':m.free,'used_bytes':m.used,'reserved_bytes':m.reserved}
worker=None
def cleanup():
 global worker
 if worker is not None:
  worker.terminate()
  try:worker.wait(timeout=20)
  except subprocess.TimeoutExpired:worker.kill();worker.wait()
  worker=None
atexit.register(cleanup)
cfg=e.GenerateConfig(pretrained_checkpoint=str(V/'models/openvla-oft-spatial'),seed=42,use_wandb=False)
e.validate_config(cfg);e.set_seed_everywhere(42)
model,head,proprio,noisy,processor=e.initialize_model(cfg);size=e.get_image_resize_size(cfg)
suite=e.benchmark.get_benchmark_dict()['libero_spatial']();observations=[]
# Identical fixed official task initial state 0; simulator excluded from measurement.
for i in range(10):
 task=suite.get_task(i);env,desc=e.get_libero_env(task,cfg.model_family,resolution=256)
 env.reset();obs=env.set_init_state(suite.get_task_init_states(i)[0])
 for _ in range(cfg.num_steps_wait):obs,*_=env.step(e.get_libero_dummy_action(cfg.model_family))
 observations.append((obs,desc));env.close()
with open(out/'observations.pkl','wb') as f:pickle.dump(observations,f,protocol=4)
original=e.get_libero_image
kind='raw'
def image(obs):
 im=original(obs)
 if worker is not None:
  pickle.dump({'shape':im.shape,'bytes':im.tobytes()},worker.stdin,protocol=4);worker.stdin.flush();reply=pickle.load(worker.stdout)
  im=np.frombuffer(reply['bytes'],dtype=np.uint8).reshape(reply['shape']).copy()
 return im
e.get_libero_image=image
def cycle(obs,desc):
 # Match current rollout implementation: prepare each of 8 control observations,
 # one VLA action chunk per 8 steps. Repeated fixed observation, no simulation.
 for i in range(cfg.num_open_loop_steps):
  observation,_=e.prepare_observation(obs,size)
  if i==0:
   actions=e.get_action(cfg,model,observation,desc,processor=processor,action_head=head,proprio_projector=proprio,noisy_action_projector=noisy,use_film=cfg.use_film)
 for a in actions:e.process_action(np.array(a,copy=True),cfg.model_family)
records=[];kinds=['raw','vae','42','43','44']
protocol={'scope':'fixed-observation replay, 8 observation preprocessing calls + 1 action-chunk inference + action postprocessing; excludes simulator/model startup','shared_gpu':shared,'acceptance_eligible':not shared,'energy_scope':'whole GPU board; shared runs include other processes','rounds':3,'tasks':10,'cycles_per_task':3,'warmup_cycles_per_condition':3,'order':'cyclic rotation per round','conditions':kinds,'observation_sha256':hashlib.sha256((out/'observations.pkl').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'confidence_intervals':False}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
for block in range(3):
 for kind in kinds[block:]+kinds[:block]:
  guard();cleanup()
  if kind!='raw':
   worker=subprocess.Popen([str(R/'venv/bin/python'),'-u',str(V/'reconstruction_worker.py'),kind],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=open(out/f'worker_{kind}.log','a'))
   assert pickle.load(worker.stdout)['ready']
  for _ in range(3):cycle(*observations[0])
  times=[];before=guard();torch.cuda.synchronize();energy0=nv.nvmlDeviceGetTotalEnergyConsumption(h);start=time.perf_counter()
  for obs,desc in observations:
   for _ in range(3):
    guard();torch.cuda.synchronize();t=time.perf_counter();cycle(obs,desc);torch.cuda.synchronize();times.append(time.perf_counter()-t)
  elapsed=time.perf_counter()-start;energy1=nv.nvmlDeviceGetTotalEnergyConsumption(h);after=guard()
  rec={'block':block,'kind':kind,'cycle_seconds':times,'mean_seconds':float(np.mean(times)),'p95_seconds':float(np.percentile(times,95)),'board_joules':(energy1-energy0)/1000,'joules_per_cycle':(energy1-energy0)/1000/len(times),'energy_window_seconds':elapsed,'before':before,'after':after,'shared_gpu':shared,'acceptance_eligible':not shared}
  records.append(rec)
  with open(out/'blocks.jsonl','a') as f:f.write(json.dumps(rec)+'\n')
  (out/'progress.json').write_text(json.dumps({'completed':len(records),'total':15,'kind':kind,'block':block,'updated':time.time()}))
  print('BLOCK_COMPLETE',block,kind,rec['mean_seconds'],flush=True)
cleanup();(out/'result.json').write_text(json.dumps({'protocol':protocol,'blocks':records},indent=2))
