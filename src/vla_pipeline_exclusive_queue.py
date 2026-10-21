"""Run matched pipeline benchmark only on an idle, unshared GPU."""
import os,time,json,subprocess,fcntl,signal
from pathlib import Path
import pynvml as n
import psutil
R=Path('/datasets2/dzy/snn_vae_20261015');n.nvmlInit()
lock=open(R/'vla_pipeline_exclusive.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state=R/'vla_pipeline_exclusive_status.json';stable={}
def write(d):
 t=state.with_suffix('.tmp');t.write_text(json.dumps(dict(d,updated=time.time())));os.replace(t,state)
def foreign(h,owned=()):
 compute={p.pid for p in n.nvmlDeviceGetComputeRunningProcesses(h) if p.pid not in owned}
 graphics=set()
 for p in n.nvmlDeviceGetGraphicsRunningProcesses(h):
  if p.pid in owned:continue
  try:name=psutil.Process(p.pid).name().lower()
  except psutil.NoSuchProcess:continue
  if name not in ('xorg','xwayland'):graphics.add(p.pid)
 return sorted(compute|graphics)
while True:
 gpu=None;observations=[]
 for i in range(n.nvmlDeviceGetCount()):
  h=n.nvmlDeviceGetHandleByIndex(i);m=n.nvmlDeviceGetMemoryInfo(h,version=n.nvmlMemory_v2);u=n.nvmlDeviceGetUtilizationRates(h).gpu;other=foreign(h)
  ok=not other and m.used<256*1024**2 and m.free>=28000*1024**2 and u<=5
  stable[i]=stable.get(i,0)+1 if ok else 0
  observations.append({'gpu':i,'used_mb':m.used/1024**2,'reserved_mb':m.reserved/1024**2,'foreign_pids':other,'utilization':u})
  if stable[i]>=3 and gpu is None:gpu=i
 if gpu is None:
  write({'status':'waiting_for_exclusive_gpu','gpus':observations});time.sleep(10);continue
 out=R/'vla/runs/pipeline_exclusive'/str(time.time_ns());out.mkdir(parents=True)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',EXPERIMENT_SHARED_GPU='0',PIPELINE_OUTPUT=str(out),LIBERO_CONFIG_PATH=str(R/'vla/libero-config'),MUJOCO_GL='egl',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
 h=n.nvmlDeviceGetHandleByIndex(gpu)
 (out/'gpu_identity.json').write_text(json.dumps({'gpu':gpu,'uuid':n.nvmlDeviceGetUUID(h),'shared_gpu':False}))
 interrupted=False
 with open(R/'logs/vla_pipeline_exclusive.log','a') as log:
  p=subprocess.Popen([str(R/'vla/venv/bin/python'),'-u',str(R/'vla/benchmark_pipeline.py')],env=env,cwd=R/'vla/vendor/openvla-oft',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write({'status':'running','pid':p.pid,'gpu':gpu,'output':str(out)})
  while p.poll() is None:
   time.sleep(2)
   try:owned={p.pid}|{c.pid for c in psutil.Process(p.pid).children(recursive=True)}
   except psutil.NoSuchProcess:break
   other=foreign(h,owned)
   if other and p.poll() is None:
    (out/'invalidated.json').write_text(json.dumps({'reason':'foreign GPU process','pids':other,'time':time.time()}));os.killpg(p.pid,signal.SIGTERM);interrupted=True;break
  code=p.wait()
 (out/'exit_status.json').write_text(json.dumps({'exit_code':code,'invalidated':interrupted}))
 if interrupted:stable={};continue
 write({'status':'complete' if code==0 else 'failed','exit_code':code,'gpu':gpu,'output':str(out)});break
