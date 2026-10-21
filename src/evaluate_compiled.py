"""Validation comparison and exclusive paired performance screening; not final test."""
import hashlib,json,os,signal,sys,time
from pathlib import Path
import numpy as np
import torch
import lpips
import pynvml as nv
from torch.utils.data import DataLoader
from common import config,seed_all,teacher,Frames,write_json
import run
from spike_encoder import SpikeEncoder
from analog_encoder import AnalogEncoder
STOP=False
def stop(*args):
 global STOP
 STOP=True;run.STOP=True
signal.signal(signal.SIGTERM,stop)

@torch.inference_mode()
def main():
 cfg=config();seed_all(cfg['seed']);root=Path(cfg['root']);mode=sys.argv[1]
 if os.environ.get('EXPERIMENT_SHARED_GPU')=='1':
  torch.cuda.set_per_process_memory_fraction(cfg['shared_training_policy']['memory_fraction'])
 reference=teacher(cfg,'cuda');original={k:v.detach().cpu().clone() for k,v in reference.decoder.state_dict().items()}
 models={};saved={};hashes={}
 for kind,folder,cls in [('snn','joint_v4',SpikeEncoder),('ann','ann_matched_v4',AnalogEncoder)]:
  path=root/'runs'/folder/'best.pt';hashes[kind]=hashlib.sha256(path.read_bytes()).hexdigest()
  ck=torch.load(path,map_location='cpu',weights_only=False);spec=ck['spec'];m=cls(spec['width'],spec['depth'],spec['timesteps']).cuda().eval();m.load_state_dict(ck['model']);m.set_blend(1)
  models[kind]=m;saved[kind]=ck
 def select(kind):
  reference.decoder.load_state_dict(original if kind=='baseline' else saved[kind]['decoder'])
  def fn(x):
   z=reference.encode(x).latent_dist.mode() if kind=='baseline' else models[kind](x)[0]
   return reference.decode(z).sample
  return fn
 out=root/'runs/candidate_comparison_compiled';out.mkdir(parents=True,exist_ok=True)
 if mode=='quality':
  fn=select('ann');metric=lpips.LPIPS(net='alex').cuda().eval();rows=[]
  for x,paths in DataLoader(Frames(cfg,'validation'),batch_size=4):
   if STOP:raise SystemExit(75)
   started=time.perf_counter();x=x.cuda();raw=fn(x)
   if not torch.isfinite(raw).all():raise RuntimeError('Nonfinite output')
   y=raw.clamp(-1,1)
   scores={'psnr':-10*torch.log10(((x-y)/2).square().mean((1,2,3)).clamp_min(1e-12)),'ssim':run.ssim((x+1)/2,(y+1)/2),'lpips':metric(y,x).flatten()}
   rows.extend(dict(path=p,**{k:v[i].item() for k,v in scores.items()}) for i,p in enumerate(paths))
   if len(rows)%400==0:print(json.dumps(dict(event='ann_validation',frames=len(rows))),flush=True)
   torch.cuda.synchronize();time.sleep((time.perf_counter()-started)*cfg['shared_training_policy']['pause_to_work_ratio'])
  result=dict(kind='ann',split='validation',frames=rows,quality={k:float(np.mean([r[k] for r in rows])) for k in scores},checkpoint_sha256=hashes['ann'],performance=None,stage1_complete=False)
  write_json(out/'ann_full.json',result)
  source=root/'runs/joint_v4/audit'
  snn=json.loads((source/'snn_full.json').read_text());base=json.loads((source/'baseline_full.json').read_text())
  assert hashes['snn']==snn['checkpoint_sha256']
  assert [r['path'] for r in rows]==[r['path'] for r in snn['frames']]==[r['path'] for r in base['frames']]
  summary={k:r['quality'] for k,r in [('baseline',base),('snn',snn),('ann',result)]}
  write_json(out/'quality_summary.json',dict(quality=summary,checkpoint_hashes=hashes,stage1_complete=False))
  print(json.dumps(summary),flush=True)
 elif mode=='performance':
  if os.environ.get('EXPERIMENT_SHARED_GPU')=='1':raise RuntimeError('Exclusive GPU required')
  nv.nvmlInit();handle=nv.nvmlDeviceGetHandleByUUID(os.environ['EXPERIMENT_NVML_UUID']);run.exclusive(handle)
  dataset=Frames(cfg,'validation');x=dataset[0][0].unsqueeze(0).cuda();blocks=[];compiled={};equivalence={}
  os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(root/'torchinductor-cache'))
  # Balanced cyclic orders: each model occupies every position once.
  for block,order in enumerate([['baseline','snn','ann'],['snn','ann','baseline'],['ann','baseline','snn']]):
   measurements={}
   for kind in order:
    if STOP:raise SystemExit(75)
    eager=select(kind);torch.cuda.synchronize();run.exclusive(handle)
    if kind not in compiled:
     started=time.perf_counter();compiled[kind]=torch.compile(eager,mode='reduce-overhead',fullgraph=True)
     errors=[]
     for idx in range(16):
      inp=dataset[idx][0].unsqueeze(0).cuda();expected=eager(inp).clone();actual=compiled[kind](inp).clone()
      error=(expected-actual).abs().max().item();errors.append(error)
      if not torch.isfinite(actual).all() or error>0.001:raise RuntimeError(f'Compiled equivalence failed: {kind} {error}')
     torch.cuda.synchronize();equivalence[kind]=dict(max_abs_error=max(errors),validation_images=16,compilation_and_check_seconds=time.perf_counter()-started)
     print(json.dumps(dict(event='compiled_verified',kind=kind,**equivalence[kind])),flush=True)
    fn=compiled[kind]
    perf=run.benchmark(fn,x,handle);measurements[kind]=perf
    print(json.dumps(dict(event='performance_block',block=block,kind=kind,wall_p95_ms=perf['wall_p95_ms'],energy=[r['gpu_joules_per_image'] for r in perf['rounds']])),flush=True)
   blocks.append(dict(block=block,order=order,measurements=measurements))
  run.exclusive(handle)
  write_json(out/'paired_performance.json',dict(checkpoint_hashes=hashes,blocks=blocks,compilation=equivalence,gpu_uuid=os.environ['EXPERIMENT_NVML_UUID'],gpu_name=torch.cuda.get_device_name(),stage1_complete=False,scope='All models torch.compile reduce-overhead fullgraph; excludes compilation; validation screening, resident batch1 FP32 model forward only; no test data; shared resident model memory is not isolated per-model memory',next_required='Independent frozen test, multiple training seeds, fine-tuned original baseline control, confidence intervals, full VLA costs'))
 else:raise ValueError(mode)
if __name__=='__main__':main()
