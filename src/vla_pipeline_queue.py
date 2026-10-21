import os,time,json,subprocess,fcntl
from pathlib import Path
import pynvml as n
R=Path('/datasets2/dzy/snn_vae_20261015');n.nvmlInit();h=n.nvmlDeviceGetHandleByIndex(3)
lock=open(R/'vla_pipeline_queue.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state=R/'vla_pipeline_status.json'
while True:
 m=n.nvmlDeviceGetMemoryInfo(h,version=n.nvmlMemory_v2);u=n.nvmlDeviceGetUtilizationRates(h).gpu
 available=[]
 for i in (2,3):
  hh=n.nvmlDeviceGetHandleByIndex(i);mm=n.nvmlDeviceGetMemoryInfo(hh,version=n.nvmlMemory_v2);uu=n.nvmlDeviceGetUtilizationRates(hh).gpu
  if mm.free>=28000*1024**2 and uu<=50:available.append((mm.free,i))
 if available:
  gpu=max(available)[1];break
 state.write_text(json.dumps({'status':'waiting_for_gpu3_headroom','free_mb':m.free/1024**2,'required_free_mb':28000,'utilization':u,'updated':time.time()}));time.sleep(10)
out=R/f'vla/runs/pipeline_shared_gpu{gpu}'/str(time.time_ns());out.mkdir(parents=True)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',EXPERIMENT_SHARED_GPU='1',PIPELINE_OUTPUT=str(out),LIBERO_CONFIG_PATH=str(R/'vla/libero-config'),MUJOCO_GL='egl',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
with open(R/f'logs/vla_pipeline_shared_gpu{gpu}.log','a') as log:
 p=subprocess.Popen([str(R/'vla/venv/bin/python'),'-u',str(R/'vla/benchmark_pipeline.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
 state.write_text(json.dumps({'status':'running','pid':p.pid,'gpu':gpu,'output':str(out),'shared_gpu':True}));code=p.wait()
state.write_text(json.dumps({'status':'complete' if code==0 else 'failed','exit_code':code,'output':str(out),'shared_gpu':True,'updated':time.time()}))
