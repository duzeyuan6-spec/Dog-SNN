"""Validation-only numerical gate for a new, separately versioned memory layout."""
import copy,hashlib,json
from pathlib import Path
import torch
from common import config,seed_all,teacher,Frames,write_json
from spike_encoder import SpikeEncoder
from spike_encoder_fast import specialize
@torch.inference_mode()
def main():
 cfg=config();seed_all(42);root=Path(cfg['root']);out=root/'runs/decoder_layout_probe';out.mkdir(exist_ok=True)
 manifest=json.loads((root/'runs/frozen_t1/manifest.json').read_text())
 for rel,h in manifest['files'].items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==h
 torch.cuda.set_per_process_memory_fraction(.35);dataset=Frames(cfg,'validation');results={}
 for kind in ['baseline','42','43','44']:
  ref=teacher(cfg,'cuda');m=None
  if kind!='baseline':
   ck=torch.load(root/manifest['checkpoints'][kind],map_location='cpu',weights_only=False);sp=ck['spec']
   m=SpikeEncoder(sp['width'],sp['depth'],sp['timesteps']).cuda().eval();m.load_state_dict(ck['model']);m.set_blend(1);specialize(m);ref.decoder.load_state_dict(ck['decoder'])
  alt=copy.deepcopy(ref);alt.decoder.to(memory_format=torch.channels_last)
  am=copy.deepcopy(m) if m is not None else None
  errors=[]
  for i in range(16):
   x=dataset[i][0].unsqueeze(0).cuda();xc=x
   z=ref.encode(x).latent_dist.mode() if m is None else m(x)[0]
   az=alt.encode(xc).latent_dist.mode() if am is None else am(xc)[0]
   a=ref.decode(z).sample;b=alt.decode(az).sample
   assert torch.isfinite(a).all() and torch.isfinite(b).all()
   errors.append(float((a-b).abs().max()))
  results[kind]=dict(max_abs=max(errors),per_frame_max=errors,numerical_pass=max(errors)<=.001)
  write_json(out/'numerical_gate.json',dict(split='validation',threshold=.001,results=results,stage1_complete=False,performance=None))
  print(json.dumps(dict(kind=kind,**results[kind])),flush=True)
  del ref,alt,m,am;torch.cuda.empty_cache()
if __name__=='__main__':main()
