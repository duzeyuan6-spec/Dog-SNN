"""Validation only: full validation and causal pulse ablations; test sealed."""
import hashlib,json,os,signal,time
from pathlib import Path
import numpy as np
import torch
import lpips
from torch.utils.data import DataLoader
from common import config,seed_all,teacher,Frames,write_json
from run import ssim
from spike_encoder import SpikeEncoder,RateIF
STOP=False
def stop(*args):
 global STOP
 STOP=True
signal.signal(signal.SIGTERM,stop)

@torch.inference_mode()
def main():
 cfg=config();seed_all(cfg['seed'])
 torch.cuda.set_per_process_memory_fraction(cfg['shared_training_policy']['memory_fraction'])
 root=Path(cfg['root']);out=root/'runs/joint_v4/audit';out.mkdir(parents=True,exist_ok=True)
 path=root/'runs/joint_v4/best.pt';sha=hashlib.sha256(path.read_bytes()).hexdigest()
 saved=torch.load(path,map_location='cpu',weights_only=False)
 spec=saved['spec'];model=SpikeEncoder(spec['width'],spec['depth'],spec['timesteps']).cuda().eval();model.load_state_dict(saved['model']);model.set_blend(1)
 reference=teacher(cfg,'cuda');original={k:v.detach().cpu().clone() for k,v in reference.decoder.state_dict().items()}
 metric=lpips.LPIPS(net='alex').cuda().eval()
 for mode,limit in [('baseline_full',None),('snn_full',None),('analog_same_weights',400),('zero_pulses',400)]:
  dest=out/(mode+'.json')
  if dest.exists():
   previous=json.loads(dest.read_text())
   if previous['checkpoint_sha256']!=sha: raise RuntimeError('Checkpoint changed during audit')
   continue
  reference.decoder.load_state_dict(original if mode=='baseline_full' else saved['decoder'])
  hooks=[]
  model.set_blend(0 if mode=='analog_same_weights' else 1)
  if mode=='zero_pulses':
   for m in model.modules():
    if isinstance(m,RateIF):hooks.append(m.register_forward_hook(lambda module,args,result:torch.zeros_like(result)))
  rows=[];audit=[]
  for m in model.modules():
   if isinstance(m,RateIF):m.audit=(mode=='snn_full')
  for x,paths in DataLoader(Frames(cfg,'validation',limit),batch_size=4):
   if STOP:raise SystemExit(75)
   started=time.perf_counter();x=x.cuda()
   mu=reference.encode(x).latent_dist.mode() if mode=='baseline_full' else model(x)[0]
   raw=reference.decode(mu).sample
   if not torch.isfinite(raw).all():raise RuntimeError('Nonfinite reconstruction')
   y=raw.clamp(-1,1)
   scores={'psnr':-10*torch.log10(((x-y)/2).square().mean((1,2,3)).clamp_min(1e-12)), 'ssim':ssim((x+1)/2,(y+1)/2),'lpips':metric(y,x).flatten()}
   rows.extend(dict(path=p,**{k:v[i].item() for k,v in scores.items()}) for i,p in enumerate(paths))
   if mode=='snn_full' and not audit:
    for m in model.modules():
     if isinstance(m,RateIF):audit.append(m.last_audit);m.audit=False
    assert all(a['binary'] for a in audit)
    assert torch.equal(model(x)[0],mu),'state did not reset'
   if len(rows)%400==0:print(json.dumps(dict(mode=mode,frames=len(rows))),flush=True)
   torch.cuda.synchronize();time.sleep((time.perf_counter()-started)*cfg['shared_training_policy']['pause_to_work_ratio'])
  for h in hooks:h.remove()
  quality={k:float(np.mean([r[k] for r in rows])) for k in ('psnr','ssim','lpips')}
  write_json(dest,dict(mode=mode,split='validation',checkpoint_sha256=sha,checkpoint_step=saved['step'],frames=rows,quality=quality,spike_audit=audit,performance=None,stage1_complete=False,control_note='Analog same weights is an ablation, not a trained matched ANN control'))
  print(json.dumps(dict(event='audit_result',mode=mode,quality=quality,frames=len(rows))),flush=True)
 results={m:json.loads((out/(m+'.json')).read_text()) for m in ['baseline_full','snn_full','analog_same_weights','zero_pulses']}
 b=results['baseline_full']['quality'];s=results['snn_full']['quality']
 subset=results['snn_full']['frames'][:400]
 write_json(out/'summary.json',dict(checkpoint_sha256=sha,full_validation_quality_pass=s['psnr']>=b['psnr']-cfg['quality_gate']['psnr_drop_db'] and s['lpips']<=b['lpips']+cfg['quality_gate']['lpips_increase'],quality={m:r['quality'] for m,r in results.items()},snn_same_400={k:float(np.mean([r[k] for r in subset])) for k in ['psnr','ssim','lpips']},stage1_complete=False,next_required='Review pulse ablation; train matched ANN and fine-tuned baseline controls; exclusive paired benchmarks; frozen configuration and independent test; multiple seeds'))
if __name__=='__main__':main()
