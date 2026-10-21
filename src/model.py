"""Compact variational students with matched ANN and LIF architectures.

Continuous input/output and Gaussian latent heads are retained. This is a
hybrid SNN VAE, not a fully spiking probabilistic sampler. State resets per
image batch; no input-to-output shortcut or stale temporal state is used.
"""
import torch
from torch import nn
from torch.nn import functional as F


class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad / (1 + 5 * x.abs()).square()


class TemporalActivation(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind

    def forward(self, x):
        if self.kind == 'ann':
            return F.silu(x)
        membrane = torch.zeros_like(x[0])
        outputs = []
        for current in x.unbind(0):
            membrane = 0.5 * membrane + current
            spike = Spike.apply(membrane - 1.0)
            membrane = membrane - spike.detach()
            outputs.append(spike)
        return torch.stack(outputs)


class Block(nn.Module):
    def __init__(self, cin, cout, kind, stride=1, up=False):
        super().__init__()
        self.up = up
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.norm = nn.GroupNorm(8, cout)
        self.act = TemporalActivation(kind)

    def forward(self, x):
        t, b = x.shape[:2]
        y = x.flatten(0, 1)
        if self.up:
            y = F.interpolate(y, scale_factor=2, mode='nearest')
        y = self.norm(self.conv(y))
        return self.act(y.unflatten(0, (t, b)))


class StudentVAE(nn.Module):
    def __init__(self, kind='snn', width=48, timesteps=4, latent_channels=32):
        super().__init__()
        if kind not in ('snn', 'ann'):
            raise ValueError(kind)
        self.kind = kind
        self.timesteps = timesteps if kind == 'snn' else 1
        self.stem = nn.Conv2d(3, width, 3, padding=1)
        self.encoder = nn.Sequential(
            Block(width, width, kind, 2), Block(width, width, kind),
            Block(width, width * 2, kind, 2), Block(width * 2, width * 2, kind),
            Block(width * 2, width * 4, kind, 2), Block(width * 4, width * 4, kind))
        self.moments = nn.Conv2d(width * 4, latent_channels * 2, 1)
        self.latent_in = nn.Conv2d(latent_channels, width * 4, 3, padding=1)
        self.decoder = nn.Sequential(
            Block(width * 4, width * 4, kind), Block(width * 4, width * 2, kind, up=True),
            Block(width * 2, width * 2, kind), Block(width * 2, width, kind, up=True),
            Block(width, width, kind), Block(width, width, kind, up=True))
        self.readout = nn.Conv2d(width, 3, 3, padding=1)

    def encode(self, x):
        currents = self.stem(x).unsqueeze(0).expand(self.timesteps, -1, -1, -1, -1)
        mu, logvar = self.moments(self.encoder(currents).mean(0)).chunk(2, dim=1)
        return mu, logvar.clamp(-20, 10)

    def decode(self, z):
        currents = self.latent_in(z).unsqueeze(0).expand(self.timesteps, -1, -1, -1, -1)
        return self.readout(self.decoder(currents).mean(0)).tanh()

    def forward(self, x, sample=False):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if sample else mu
        return self.decode(z), mu, logvar
