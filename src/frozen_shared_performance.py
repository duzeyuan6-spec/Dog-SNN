"""Predeclared 12-block paired frozen model performance acceptance."""
import os,json,time,hashlib,signal
from pathlib import Path
import numpy as np
import torch
import pynvml as nv
from common import config,seed_all,teacher,Frames,write_json
from spike_encoder import SpikeEncoder
from spike_encoder_fast import specialize
import run
import shared_trial as diagnostic
STOP=False
def stop(*_):
 global STOP
 STOP=True;run.STOP=True;diagnostic.STOP=True
signal.signal(signal.SIGTERM,stop)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
@torch.inference_mode()
def main():
 cfg=config();root=Path(cfg['root']);out=root/'runs/frozen_t1';seed_all(42)
 manifest=json.loads((out/'manifest.json').read_text());protocol=json.loads((out/'shared_performance_protocol.json').read_text())
 assert sha(Path(__file__))==protocol['script_sha256']
 assert sha(out/'manifest.json')==protocol['manifest_sha256']
 for rel,h in manifest['files'].items():assert sha(root/rel)==h,rel
 assert os.environ.get('EXPERIMENT_SHARED_GPU')=='1'
 torch.cuda.set_per_process_memory_fraction(.35)
 qualities={k:json.loads((out/(k+'_quality.json')).read_text()) for k in ['baseline','42','43','44']}
 base=qualities['baseline'];paths=[r['path'] for r in base['frames']]
 for k,q in qualities.items():
  assert q['manifest_sha256']==protocol['manifest_sha256']
  assert [r['path'] for r in q['frames']]==paths
  assert q['quality']['psnr']>=base['quality']['psnr']-cfg['quality_gate']['psnr_drop_db']
  assert q['quality']['lpips']<=base['quality']['lpips']+cfg['quality_gate']['lpips_increase']
 nv.nvmlInit();handle=nv.nvmlDeviceGetHandleByUUID(os.environ['EXPERIMENT_NVML_UUID'])
 diagnostic.INITIAL=set(diagnostic.foreign_processes(handle,{os.getpid()}))
 diagnostic.guard(handle)
 out=out/'shared_performance'/os.environ['TRIAL_GPU'];out.mkdir(parents=True,exist_ok=True)
 write_json(out/'occupancy_start.json',dict(time=time.time(),foreign_pids=sorted(diagnostic.INITIAL),gpu_uuid=os.environ['EXPERIMENT_NVML_UUID'],acceptance_eligible=False))
 reference=teacher(cfg,'cuda');original={k:v.cpu().clone() for k,v in reference.decoder.state_dict().items()};models={};decoders={}
 for seed,path in manifest['checkpoints'].items():
  ck=torch.load(root/path,map_location='cpu',weights_only=False);sp=ck['spec']
  m=SpikeEncoder(sp['width'],sp['depth'],sp['timesteps']).cuda().eval();m.load_state_dict(ck['model']);m.set_blend(1);specialize(m);models[seed]=m;decoders[seed]=ck['decoder']
 dataset=Frames(cfg,'test');attempt=out/'performance_attempts'/str(time.time_ns());attempt.mkdir(parents=True)
 blocks=[];names=['baseline','42','43','44']
 for block in range(12):
  order=names[block%4:]+names[:block%4];index=protocol['test_indices'][block];x=dataset[index][0].unsqueeze(0).cuda();measurements={}
  for kind in order:
   if STOP:raise SystemExit(75)
   diagnostic.guard(handle);reference.decoder.load_state_dict(original if kind=='baseline' else decoders[kind])
   def fn(x):
    z=reference.encode(x).latent_dist.mode() if kind=='baseline' else models[kind](x)[0]
    return reference.decode(z).sample
   perf=diagnostic.benchmark(fn,x,handle);measurements[kind]=perf
   write_json(out/f'occupancy_{block}_{kind}.json',dict(time=time.time(),foreign_pids=sorted(diagnostic.foreign_processes(handle,{os.getpid()})),util=nv.nvmlDeviceGetUtilizationRates(handle).gpu))
   print(json.dumps(dict(block=block+1,total_blocks=12,kind=kind,p95_ms=perf['wall_p95_ms'],joules=[r['gpu_joules_per_image'] for r in perf['rounds']])),flush=True)
   write_json(out/'performance_progress.json',dict(block=block+1,kind=kind,completed_measurements=block*4+len(measurements),total_measurements=48,updated_at=time.time(),attempt=str(attempt)))
  blocks.append(dict(block=block,order=order,test_index=index,measurements=measurements));write_json(attempt/f'block_{block:02}.json',blocks[-1])
 diagnostic.guard(handle)
 values={k:np.array([[b['measurements'][k]['wall_p95_ms'],np.mean([r['gpu_joules_per_image'] for r in b['measurements'][k]['rounds']])] for b in blocks]) for k in names}
 rng=np.random.default_rng(20261020);indices=rng.integers(0,12,(10000,12));summary={}
 for k in names[1:]:
  reductions=1-values[k].mean(0)/values['baseline'].mean(0)
  samples=1-values[k][indices].mean(1)/values['baseline'][indices].mean(1);ci=np.percentile(samples,[2.5,97.5],axis=0)
  summary[k]=dict(mean_p95_ms=float(values[k][:,0].mean()),mean_board_joules=float(values[k][:,1].mean()),reductions=reductions.tolist(),paired_block_bootstrap95=ci.tolist(),performance_pass=None,shared_screening_meets_numeric_threshold=bool(np.all(ci[0]>=.1)))
 write_json(out/'paired_performance.json',dict(blocks=blocks,summary=summary,baseline_means=values['baseline'].mean(0).tolist(),gpu_uuid=os.environ['EXPERIMENT_NVML_UUID'],protocol=protocol,stage1_complete=False,shared_gpu=True,acceptance_eligible=False,scope='Shared GPU resident batch1 forward; whole-board energy INCLUDES other users processes, not isolated SNN energy',remaining='Exclusive confirmation plus quality uncertainty and posterior/cost review before VLA'))
 print(json.dumps(dict(event='frozen_performance_complete',summary=summary)),flush=True)
if __name__=='__main__':main()
