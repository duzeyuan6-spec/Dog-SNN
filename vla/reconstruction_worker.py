"""Local pipe worker using the frozen VAE environment; stdout is binary IPC only."""
import sys,pickle,hashlib,json
from pathlib import Path
ROOT=Path('/datasets2/dzy/snn_vae_20261015');sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from common import config,teacher,seed_all
from spike_encoder import SpikeEncoder
from spike_encoder_fast import specialize
seed_all(42);cfg=config();kind=sys.argv[1]
manifest=json.loads((ROOT/'runs/frozen_t1/manifest.json').read_text())
for rel,h in manifest['files'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h
ref=teacher(cfg,'cuda');model=None
if kind!='vae':
 ck=torch.load(ROOT/manifest['checkpoints'][kind],map_location='cpu',weights_only=False);sp=ck['spec'];model=SpikeEncoder(sp['width'],sp['depth'],sp['timesteps']).cuda().eval();model.load_state_dict(ck['model']);model.set_blend(1);specialize(model);ref.decoder.load_state_dict(ck['decoder'])
ref.decoder.to(memory_format=torch.channels_last)
pickle.dump({'ready':True},sys.stdout.buffer,protocol=4);sys.stdout.buffer.flush()
with torch.inference_mode():
 while True:
  try:req=pickle.load(sys.stdin.buffer)
  except EOFError:break
  a=np.frombuffer(req['bytes'],dtype=np.uint8).reshape(req['shape']).copy();assert a.shape==(256,256,3)
  x=torch.from_numpy(a).permute(2,0,1).unsqueeze(0).cuda().float()/127.5-1
  z=ref.encode(x).latent_dist.mode() if model is None else model(x)[0]
  y=((ref.decode(z).sample.clamp(-1,1)+1)*127.5).round().byte()[0].permute(1,2,0).cpu().numpy()
  pickle.dump({'shape':y.shape,'bytes':y.tobytes()},sys.stdout.buffer,protocol=4);sys.stdout.buffer.flush()
