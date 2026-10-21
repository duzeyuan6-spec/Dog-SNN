"""Validation-only decoder channels-last evaluation; frozen artifacts remain unchanged."""
import os,sys,json,time,hashlib,signal
from pathlib import Path
import torch,lpips,numpy as np
from torch.utils.data import DataLoader
from common import config,seed_all,teacher,Frames,write_json
from spike_encoder import SpikeEncoder
from spike_encoder_fast import specialize
from run import ssim
STOP=False
def stop(*_):
 global STOP
 STOP=True
signal.signal(signal.SIGTERM,stop)
@torch.inference_mode()
def main():
 cfg=config();root=Path(cfg['root']);kind=sys.argv[1];seed_all(42)
 manifest=json.loads((root/'runs/frozen_t1/manifest.json').read_text())
 for rel,sha in manifest['files'].items():
  assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==sha, 'Frozen artifact changed: '+rel
 torch.cuda.set_per_process_memory_fraction(.35)
 ref=teacher(cfg,'cuda');model=None
 if kind!='baseline':
  path=root/manifest['checkpoints'][kind]
  ck=torch.load(path,map_location='cpu',weights_only=False);sp=ck['spec']
  model=SpikeEncoder(sp['width'],sp['depth'],sp['timesteps']).cuda().eval();model.load_state_dict(ck['model']);model.set_blend(1);specialize(model);ref.decoder.load_state_dict(ck['decoder'])
 ref.decoder.to(memory_format=torch.channels_last)
 gate=json.loads((root/'runs/decoder_layout_probe/numerical_gate.json').read_text())
 assert all(v['numerical_pass'] for v in gate['results'].values()) and len(gate['results'])==4
 metric=lpips.LPIPS(net='alex').cuda().eval();rows=[]
 for x,paths in DataLoader(Frames(cfg,'validation'),batch_size=4):
  if STOP:raise SystemExit(75)
  started=time.perf_counter();x=x.cuda();z=ref.encode(x).latent_dist.mode() if model is None else model(x)[0];raw=ref.decode(z).sample
  assert torch.isfinite(raw).all();y=raw.clamp(-1,1)
  scores={'psnr':-10*torch.log10(((x-y)/2).square().mean((1,2,3)).clamp_min(1e-12)),'ssim':ssim((x+1)/2,(y+1)/2),'lpips':metric(y,x).flatten()}
  rows.extend(dict(path=p,**{k:v[i].item() for k,v in scores.items()}) for i,p in enumerate(paths))
  if len(rows)%400==0:print(json.dumps(dict(kind=kind,validation_frames=len(rows))),flush=True)
  torch.cuda.synchronize()
  if os.environ.get('EXPERIMENT_SHARED_GPU')=='1':time.sleep((time.perf_counter()-started)*cfg['shared_training_policy']['pause_to_work_ratio'])
 out=root/'runs/decoder_layout_validation';out.mkdir(exist_ok=True);quality={k:float(np.mean([r[k] for r in rows])) for k in scores}
 write_json(out/(kind+'_quality.json'),dict(kind=kind,split='validation',frames=rows,quality=quality,performance=None,stage1_complete=False,manifest_sha256=hashlib.sha256((root/'runs/frozen_t1/manifest.json').read_bytes()).hexdigest()))
 print(json.dumps(dict(kind=kind,quality=quality)),flush=True)
if __name__=='__main__':main()
