"""Single-step inference specialization; keeps normalized-current threshold exactly."""
import torch
from spike_encoder import RateIF
from model import Spike
class FastRateIF(RateIF):
 def forward(self,x):
  if self.timesteps != 1 or self.blend != 1.0 or self.training:
   return super().forward(x)
  scale=self.log_scale.exp().clamp(0.1,10)
  current=(x/scale).clamp(0,1)
  spike=Spike.apply(current-1)
  if self.audit:
   self.last_audit={'binary':bool(((spike==0)|(spike==1)).all()),'firing_rate':spike.mean().item(),'timesteps':1}
  return scale*spike

def specialize(model):
 for m in model.modules():
  if isinstance(m,RateIF):
   assert m.timesteps==1 and m.blend==1
   m.__class__=FastRateIF
 return model
