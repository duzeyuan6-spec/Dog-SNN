"""Hybrid encoder v2: continuous residual state, binary IF residual branches.

Only the encoder is replaced. The original frozen FLUX.2 decoder remains.
Rate averaging before linear convolution is exact for these constant-current
IF branches; this is not an event-driven GPU kernel or a fully spiking VAE.
"""
import torch
from torch import nn
from torch.nn import functional as F
from model import Spike


class RateIF(nn.Module):
    def __init__(self, channels, timesteps=4):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.timesteps = timesteps
        self.blend = 1.0
        self.audit = False
        self.last_audit = None

    def forward(self, x):
        scale = self.log_scale.exp().clamp(0.1, 10)
        current = (x / scale).clamp(0, 1)
        membrane = torch.zeros_like(current)
        spikes = []
        for _ in range(self.timesteps):
            membrane = membrane + current
            spike = Spike.apply(membrane - 1)
            membrane = membrane - spike.detach()
            spikes.append(spike)
        rate = torch.stack(spikes).mean(0)
        if self.audit:
            s = torch.stack(spikes).detach()
            self.last_audit = {'binary': bool(((s == 0) | (s == 1)).all()),
                               'firing_rate': s.mean().item(), 'timesteps': self.timesteps}
        return scale * ((1 - self.blend) * current + self.blend * rate)


class Residual(nn.Module):
    def __init__(self, width, timesteps):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, width)
        self.spike1 = RateIF(width, timesteps)
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, width)
        self.spike2 = RateIF(width, timesteps)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)

    def forward(self, x):
        y = self.conv1(self.spike1(self.norm1(x)))
        return x + 0.2 * self.conv2(self.spike2(self.norm2(y)))


class SpikeEncoder(nn.Module):
    def __init__(self, width=128, depth=6, timesteps=4):
        super().__init__()
        self.stem = nn.Conv2d(48, width, 3, padding=1)
        self.high = nn.Sequential(*(Residual(width, timesteps) for _ in range(2)))
        self.down = nn.Conv2d(width, width, 3, stride=2, padding=1)
        self.low = nn.Sequential(*(Residual(width, timesteps) for _ in range(depth-2)))
        self.head = nn.Conv2d(width, 64, 3, padding=1)

    def set_blend(self, blend):
        for module in self.modules():
            if isinstance(module, RateIF):
                module.blend = blend

    def forward(self, x):
        y = self.low(self.down(self.high(self.stem(F.pixel_unshuffle(x, 4)))))
        mu, logvar = self.head(y).chunk(2, 1)
        return mu, logvar.clamp(-30, 10)
