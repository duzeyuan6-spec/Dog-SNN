"""Validation-only decoder-layout performance screening; not final acceptance."""
import os,json,time,hashlib,signal
from pathlib import Path
import numpy as np
import torch
import pynvml as nv
from common import config,seed_all,teacher,Frames,write_json
from spike_encoder import SpikeEncoder
from spike_encoder_fast import specialize
import run
STOP=False
def stop(*_):
 global STOP
 STOP=True;run.STOP=True
signal.signal(signal.SIGTERM,stop)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
@torch.inference_mode()
def main():
 cfg=config();root=Path(cfg['root']);out=root/'runs/decoder_layout_validation';seed_all(42)
 assert not (out/'paired_performance.json').exists(), 'Do not overwrite completed run'
 manifest=json.loads((root/'runs/frozen_t1/manifest.json').read_text());protocol=json.loads((out/'performance_protocol.json').read_text())
 assert sha(Path(__file__))==protocol['script_sha256']
 assert sha(root/'runs/frozen_t1/manifest.json')==protocol['manifest_sha256']
 for rel,h in manifest['files'].items():assert sha(root/rel)==h,rel
 assert os.environ.get('EXPERIMENT_SHARED_GPU')!='1'
 qualities={k:json.loads((out/(k+'_quality.json')).read_text()) for k in ['baseline','42','43','44']}
 base=qualities['baseline'];paths=[r['path'] for r in base['frames']]
 for k,q in qualities.items():
  assert q['manifest_sha256']==protocol['manifest_sha256']
  assert [r['path'] for r in q['frames']]==paths
  assert q['quality']['psnr']>=base['quality']['psnr']-cfg['quality_gate']['psnr_drop_db']
  assert q['quality']['lpips']<=base['quality']['lpips']+cfg['quality_gate']['lpips_increase']
 nv.nvmlInit();handle=nv.nvmlDeviceGetHandleByUUID(os.environ['EXPERIMENT_NVML_UUID']);run.exclusive(handle)
 reference=teacher(cfg,'cuda');reference.decoder.to(memory_format=torch.channels_last);original={k:v.cpu().clone() for k,v in reference.decoder.state_dict().items()};models={};decoders={}
 for seed,path in manifest['checkpoints'].items():
  ck=torch.load(root/path,map_location='cpu',weights_only=False);sp=ck['spec']
  m=SpikeEncoder(sp['width'],sp['depth'],sp['timesteps']).cuda().eval();m.load_state_dict(ck['model']);m.set_blend(1);specialize(m);models[seed]=m;decoders[seed]=ck['decoder']
 dataset=Frames(cfg,'validation');attempt=out/'performance_attempts'/str(time.time_ns());attempt.mkdir(parents=True)
 blocks=[];names=['baseline','42','43','44']
 for block in range(12):
  order=names[block%4:]+names[:block%4];index=protocol['validation_indices'][block];x=dataset[index][0].unsqueeze(0).cuda();measurements={}
  for kind in order:
   if STOP:raise SystemExit(75)
   run.exclusive(handle);reference.decoder.load_state_dict(original if kind=='baseline' else decoders[kind])
   def fn(x):
    z=reference.encode(x).latent_dist.mode() if kind=='baseline' else models[kind](x)[0]
    return reference.decode(z).sample
   perf=run.benchmark(fn,x,handle);measurements[kind]=perf
   print(json.dumps(dict(block=block+1,total_blocks=12,kind=kind,p95_ms=perf['wall_p95_ms'],joules=[r['gpu_joules_per_image'] for r in perf['rounds']])),flush=True)
   write_json(out/'performance_progress.json',dict(block=block+1,kind=kind,completed_measurements=block*4+len(measurements),total_measurements=48,updated_at=time.time(),attempt=str(attempt)))
  blocks.append(dict(block=block,order=order,validation_index=index,measurements=measurements));write_json(attempt/f'block_{block:02}.json',blocks[-1])
 run.exclusive(handle)
 values={k:np.array([[b['measurements'][k]['wall_p95_ms'],np.mean([r['gpu_joules_per_image'] for r in b['measurements'][k]['rounds']])] for b in blocks]) for k in names}
 rng=np.random.default_rng(20261020);indices=rng.integers(0,12,(10000,12));summary={}
 for k in names[1:]:
  reductions=1-values[k].mean(0)/values['baseline'].mean(0)
  samples=1-values[k][indices].mean(1)/values['baseline'][indices].mean(1);ci=np.percentile(samples,[2.5,97.5],axis=0)
  summary[k]=dict(mean_p95_ms=float(values[k][:,0].mean()),mean_board_joules=float(values[k][:,1].mean()),reductions=reductions.tolist(),paired_block_bootstrap95=ci.tolist(),performance_pass=bool(np.all(ci[0]>=.1)))
 write_json(out/'paired_performance.json',dict(blocks=blocks,summary=summary,baseline_means=values['baseline'].mean(0).tolist(),gpu_uuid=os.environ['EXPERIMENT_NVML_UUID'],protocol=protocol,stage1_complete=False,acceptance_eligible=False,split='validation',scope='Resident batch1 FP32/TF32off forward; NVML whole GPU board, not full VLA or whole machine',remaining='Review quality uncertainty, posterior semantics and complete costs before VLA'))
 print(json.dumps(dict(event='decoder_validation_performance_complete',summary=summary)),flush=True)
if __name__=='__main__':main()
