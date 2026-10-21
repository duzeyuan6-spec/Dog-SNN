"""Matched nonspiking control: identical parameter shapes, continuous clipping."""
import torch
from torch import nn
from spike_encoder import SpikeEncoder as BaseEncoder, RateIF

class AnalogClip(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.log_scale = source.log_scale
    def forward(self, x):
        scale = self.log_scale.exp().clamp(0.1,10)
        return scale * (x/scale).clamp(0,1)

class AnalogEncoder(BaseEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for parent in list(self.modules()):
            for name, child in list(parent.named_children()):
                if isinstance(child, RateIF):
                    setattr(parent,name,AnalogClip(child))
    def set_blend(self, blend):
        pass
