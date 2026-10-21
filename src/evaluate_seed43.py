"""Validation comparison and exclusive paired performance screening; not final test."""
import hashlib,json,os,signal,sys,time,itertools
from pathlib import Path
import numpy as np
import torch
import lpips
import pynvml as nv
from torch.utils.data import DataLoader
from common import config,seed_all,teacher,Frames,write_json
import run
from spike_encoder import SpikeEncoder
from spike_encoder_fast import specialize
import copy
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
 for kind,folder,cls in [('snn','seed43/joint_t1_v6',SpikeEncoder),('ann','ann_matched_v4',AnalogEncoder)]:
  path=root/'runs'/folder/'best.pt';hashes[kind]=hashlib.sha256(path.read_bytes()).hexdigest()
  ck=torch.load(path,map_location='cpu',weights_only=False);spec=ck['spec'];m=cls(spec['width'],spec['depth'],spec['timesteps']).cuda().eval();m.load_state_dict(ck['model']);m.set_blend(1)
  models[kind]=m;saved[kind]=ck
 def select(kind):
  reference.decoder.load_state_dict(original if kind=='baseline' else saved[kind]['decoder'])
  def fn(x):
   z=reference.encode(x).latent_dist.mode() if kind=='baseline' else models[kind](x)[0]
   return reference.decode(z).sample
  return fn
 out=root/'runs/seed43/full_comparison';out.mkdir(parents=True,exist_ok=True)
 original_snn=copy.deepcopy(models['snn']);specialize(models['snn'])
 for j in range(16):
  probe=Frames(cfg,'validation')[j][0].unsqueeze(0).cuda()
  a=original_snn(probe);b=models['snn'](probe)
  assert all(torch.equal(u,v) for u,v in zip(a,b)), 'Specialization changed encoder output'
 del original_snn
 print(json.dumps(dict(event='specialization_exact_output_check',frames=16,passed=True)),flush=True)
 if mode=='quality':
  fn=select('snn');metric=lpips.LPIPS(net='alex').cuda().eval();rows=[]
  for x,paths in DataLoader(Frames(cfg,'validation'),batch_size=4):
   if STOP:raise SystemExit(75)
   started=time.perf_counter();x=x.cuda();raw=fn(x)
   if not torch.isfinite(raw).all():raise RuntimeError('Nonfinite output')
   y=raw.clamp(-1,1)
   scores={'psnr':-10*torch.log10(((x-y)/2).square().mean((1,2,3)).clamp_min(1e-12)),'ssim':run.ssim((x+1)/2,(y+1)/2),'lpips':metric(y,x).flatten()}
   rows.extend(dict(path=p,**{k:v[i].item() for k,v in scores.items()}) for i,p in enumerate(paths))
   if len(rows)%400==0:print(json.dumps(dict(event='snn_t1_validation',frames=len(rows))),flush=True)
   torch.cuda.synchronize();time.sleep((time.perf_counter()-started)*cfg['shared_training_policy']['pause_to_work_ratio'])
  result=dict(kind='snn',split='validation',frames=rows,quality={k:float(np.mean([r[k] for r in rows])) for k in scores},checkpoint_sha256=hashes['snn'],performance=None,stage1_complete=False)
  write_json(out/'snn_full.json',result)
  source=root/'runs/joint_v4/audit'
  ann=json.loads((root/'runs/candidate_comparison/ann_full.json').read_text());base=json.loads((source/'baseline_full.json').read_text())
  assert hashes['ann']==ann['checkpoint_sha256']
  assert [r['path'] for r in rows]==[r['path'] for r in ann['frames']]==[r['path'] for r in base['frames']]
  summary={k:r['quality'] for k,r in [('baseline',base),('snn',result),('ann',ann)]}
  write_json(out/'quality_summary.json',dict(quality=summary,checkpoint_hashes=hashes,stage1_complete=False))
  print(json.dumps(summary),flush=True)
  if not (result['quality']['psnr']>=base['quality']['psnr']-cfg['quality_gate']['psnr_drop_db'] and result['quality']['lpips']<=base['quality']['lpips']+cfg['quality_gate']['lpips_increase']):
   raise RuntimeError('Full validation quality gate failed; revise before performance test')
 elif mode=='performance':
  if os.environ.get('EXPERIMENT_SHARED_GPU')=='1':raise RuntimeError('Exclusive GPU required')
  prior=json.loads((out/'quality_summary.json').read_text())
  assert prior['checkpoint_hashes']==hashes, 'Checkpoint changed after full validation'
  nv.nvmlInit();handle=nv.nvmlDeviceGetHandleByUUID(os.environ['EXPERIMENT_NVML_UUID']);run.exclusive(handle)
  x=Frames(cfg,'validation')[0][0].unsqueeze(0).cuda();blocks=[]
  # Balanced cyclic orders: each model occupies every position once.
  attempt=str(time.time_ns()); attempt_dir=out/'repeat_attempts'/attempt; attempt_dir.mkdir(parents=True)
  orders=list(itertools.permutations(['baseline','snn','ann']))*2
  np.random.default_rng(20261018).shuffle(orders)
  for block,order in enumerate(orders):
   measurements={}
   for kind in order:
    if STOP:raise SystemExit(75)
    fn=select(kind);torch.cuda.synchronize();run.exclusive(handle)
    perf=run.benchmark(fn,x,handle);measurements[kind]=perf
    print(json.dumps(dict(event='performance_block',block=block,kind=kind,wall_p95_ms=perf['wall_p95_ms'],energy=[r['gpu_joules_per_image'] for r in perf['rounds']])),flush=True)
   blocks.append(dict(block=block,order=order,measurements=measurements))
   write_json(attempt_dir/f'block_{block:02d}.json',dict(block=blocks[-1],checkpoint_hashes=hashes,gpu_uuid=os.environ['EXPERIMENT_NVML_UUID'],complete_experiment=False))
  run.exclusive(handle)
  write_json(out/'repeated_performance.json',dict(checkpoint_hashes=hashes,blocks=blocks,gpu_uuid=os.environ['EXPERIMENT_NVML_UUID'],gpu_name=torch.cuda.get_device_name(),stage1_complete=False,scope='Validation screening, resident batch1 FP32 model forward only; no test data; shared resident model memory is not isolated per-model memory',next_required='Independent frozen test, multiple SNN training seeds, confidence intervals, full VLA costs'))
 else:raise ValueError(mode)
if __name__=='__main__':main()
