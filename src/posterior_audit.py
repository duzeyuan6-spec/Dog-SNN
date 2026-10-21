"""Validation-only common-noise posterior diagnostic, not a generative quality claim."""
import json,hashlib,time,os
from pathlib import Path
import torch
from common import config,seed_all,teacher,Frames,write_json
from spike_encoder import SpikeEncoder
from spike_encoder_fast import specialize
@torch.inference_mode()
def main():
 cfg=config();r=Path(cfg['root']);seed_all(42);torch.cuda.set_per_process_memory_fraction(.35)
 manifest=json.loads((r/'runs/frozen_t1/manifest.json').read_text())
 for rel,h in manifest['files'].items():assert hashlib.sha256((r/rel).read_bytes()).hexdigest()==h
 data=Frames(cfg,'validation');out=r/'runs/posterior_validation_audit';results={}
 for kind in ['baseline','42','43','44']:
  ref=teacher(cfg,'cuda');m=None
  if kind!='baseline':
   ck=torch.load(r/manifest['checkpoints'][kind],map_location='cpu',weights_only=False);sp=ck['spec'];m=SpikeEncoder(sp['width'],sp['depth'],sp['timesteps']).cuda().eval();m.load_state_dict(ck['model']);m.set_blend(1);specialize(m);ref.decoder.load_state_dict(ck['decoder'])
  ref.decoder.to(memory_format=torch.channels_last);rows=[]
  for i in range(16):
   start=time.perf_counter();x=data[i][0].unsqueeze(0).cuda();target=ref.encode(x).latent_dist
   mu,lv=(target.mean,target.logvar) if m is None else m(x);std=(.5*lv).exp();mode=ref.decode(mu).sample
   g=torch.Generator(device='cuda').manual_seed(20261020+i);noise=torch.randn(mu.shape,device='cuda',generator=g);sample=ref.decode(mu+std*noise).sample
   assert torch.isfinite(sample).all() and torch.isfinite(std).all()
   rows.append(dict(path=data.rows[i]['path'],mean_std=float(std.mean()),logvar_mae=float((lv-target.logvar).abs().mean()),mode_sample_mse=float((mode-sample).square().mean()),sample_psnr=float(-10*torch.log10(((x-sample.clamp(-1,1))/2).square().mean()))))
   torch.cuda.synchronize()
   if os.environ.get('EXPERIMENT_SHARED_GPU')=='1':time.sleep((time.perf_counter()-start)*cfg['shared_training_policy']['pause_to_work_ratio'])
  results[kind]=rows;write_json(out/'summary.json',dict(split='validation',results=results,performance=None,stage1_complete=False,scope='16 fixed frames, one common noise draw per frame; no claim of full generative equivalence'))
  print(json.dumps(dict(kind=kind,frames=len(rows))),flush=True);del ref,m;torch.cuda.empty_cache()
if __name__=='__main__':main()
