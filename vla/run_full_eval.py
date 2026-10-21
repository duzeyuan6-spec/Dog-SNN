"""LIBERO front-camera reconstruction adapter. Original wrist/proprio/action policy unchanged."""
import os,sys,time,json,pickle,subprocess,atexit
from pathlib import Path
import numpy as np
ROOT=Path('/datasets2/dzy/snn_vae_20261015');VLA=ROOT/'vla';sys.path.insert(0,str(VLA/'vendor/openvla-oft'));sys.path.insert(0,str(VLA/'vendor/LIBERO'))
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from experiments.robot.libero import run_libero_eval as evaluation
kind=os.environ.get('RECONSTRUCTION_KIND','raw');out=Path(os.environ['RECONSTRUCTION_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
worker=None
if kind!='raw':
 worker=subprocess.Popen([str(ROOT/'venv/bin/python'),'-u',str(VLA/'reconstruction_worker.py'),kind],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=open(out/'vae_worker.log','a'))
 assert pickle.load(worker.stdout)['ready']
 atexit.register(worker.terminate)
original=evaluation.get_libero_image
trace=open(out/'adapter_cost.jsonl','a',buffering=1)
def reconstructed(obs):
 start=time.perf_counter();im=original(obs)
 if worker is not None:
  pickle.dump({'shape':im.shape,'bytes':im.tobytes()},worker.stdin,protocol=4);worker.stdin.flush();reply=pickle.load(worker.stdout);im=np.frombuffer(reply['bytes'],dtype=np.uint8).reshape(reply['shape']).copy()
 trace.write(json.dumps({'kind':kind,'adapter_wall_seconds':time.perf_counter()-start})+'\n');return im
evaluation.get_libero_image=reconstructed
(out/'adapter_protocol.json').write_text(json.dumps({'kind':kind,'camera':'front only; wrist unchanged','resolution':256,'latent':'mode','decoder_layout':'channels_last','includes_ipc_in_adapter_cost':True,'scope':'500 episode success evaluation; shared timings descriptive, no energy claim'},indent=2))
# Measure the full observation adapter/preprocessing and VLA action calls.
import torch
cost=open(out/'pipeline_cost.jsonl','a',buffering=1)
def timed(name,fn):
 def call(*args,**kwargs):
  torch.cuda.synchronize();start=time.perf_counter();result=fn(*args,**kwargs);torch.cuda.synchronize()
  cost.write(json.dumps({'component':name,'wall_seconds':time.perf_counter()-start,'shared_gpu':os.environ.get('EXPERIMENT_SHARED_GPU')=='1'})+'\n');return result
 return call
evaluation.prepare_observation=timed('reconstruction_transfer_preprocessing',evaluation.prepare_observation)
evaluation.get_action=timed('vla_action_chunk',evaluation.get_action)
# Durable episode records for subsequent cooperative restarts; never overwrite results.
import hashlib
cache_dir=out.parent/'episode_cache';cache_dir.mkdir(exist_ok=True)
original_episode=evaluation.run_episode
original_video=evaluation.save_rollout_video
def resumable_episode(*args,**kwargs):
 task=args[2];initial=args[9]
 key=hashlib.sha256(task.encode()+np.asarray(initial).tobytes()).hexdigest();path=cache_dir/(key+'.json')
 if path.exists():return bool(json.loads(path.read_text())['success']),[]
 result=original_episode(*args,**kwargs)
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps({'task':task,'success':bool(result[0]),'attempt':str(out),'initial_sha256':key}));os.replace(tmp,path)
 return result
def save_video(images,*args,**kwargs):
 if len(images):return original_video(images,*args,**kwargs)
evaluation.run_episode=resumable_episode
evaluation.save_rollout_video=save_video
rate=evaluation.eval_libero()
(out/'result.json').write_text(json.dumps({'success_rate':rate,'kind':kind,'trials_per_task':50,'tasks':10,'confidence_interval_requested':False,'energy_measured':False},indent=2))
