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
(out/'adapter_protocol.json').write_text(json.dumps({'kind':kind,'camera':'front only; wrist unchanged','resolution':256,'latent':'mode','decoder_layout':'channels_last','includes_ipc_in_adapter_cost':True,'scope':'integration smoke; no final energy claim'},indent=2))
evaluation.eval_libero()
