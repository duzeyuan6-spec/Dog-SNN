"""Validation-only operator diagnosis after failed acceptance. No model changes."""
import hashlib,json,os
from pathlib import Path
import torch
from torch.profiler import profile,ProfilerActivity
from common import config,teacher,Frames,seed_all,write_json
from spike_encoder import SpikeEncoder
from spike_encoder_fast import specialize
@torch.inference_mode()
def main():
 cfg=config();seed_all(42);root=Path(cfg['root']);out=root/'runs/validation_operator_diagnosis';out.mkdir(exist_ok=True)
 manifest=json.loads((root/'runs/frozen_t1/manifest.json').read_text())
 for rel,h in manifest['files'].items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==h
 torch.cuda.set_per_process_memory_fraction(.35)
 ref=teacher(cfg,'cuda');original={k:v.cpu().clone() for k,v in ref.decoder.state_dict().items()}
 ck=torch.load(root/manifest['checkpoints']['42'],map_location='cpu',weights_only=False);sp=ck['spec'];m=SpikeEncoder(sp['width'],sp['depth'],sp['timesteps']).cuda().eval();m.load_state_dict(ck['model']);m.set_blend(1);specialize(m)
 x=Frames(cfg,'validation')[0][0].unsqueeze(0).cuda()
 for kind in ['baseline','snn']:
  ref.decoder.load_state_dict(original if kind=='baseline' else ck['decoder'])
  def fn():
   z=ref.encode(x).latent_dist.mode() if kind=='baseline' else m(x)[0]
   return ref.decode(z).sample
  for _ in range(10):fn()
  torch.cuda.synchronize()
  with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True) as prof:
   for _ in range(10):fn()
   torch.cuda.synchronize()
  (out/(kind+'_operators.txt')).write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))
  prof.export_chrome_trace(str(out/(kind+'_trace.json')))
  print(kind+' operator profile saved',flush=True)
 write_json(out/'summary.json',dict(split='validation',shared_gpu=os.environ.get('EXPERIMENT_SHARED_GPU')=='1',performance_claim=False,purpose='Operator diagnosis only; profiler overhead invalidates acceptance timing',frozen_artifacts_unchanged=True))
if __name__=='__main__':main()
