"""CPU contract checks; no datasets/models downloaded by this script."""
import torch
from spike_encoder import SpikeEncoder, RateIF

torch.set_num_threads(4)
torch.manual_seed(42)
model = SpikeEncoder(width=16, depth=3, timesteps=4)
model.set_blend(1)
for m in model.modules():
    if isinstance(m, RateIF):
        m.audit = True
x = torch.randn(2, 3, 32, 32)
mu, logvar = model(x)
assert mu.shape == logvar.shape == (2, 32, 4, 4)
assert torch.equal(model(x)[0], mu), 'state must reset for every batch'
(mu.square().mean()+logvar.square().mean()).backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
assert model.high[0].conv1.weight.grad.abs().sum() > 0, 'spiking branch must receive gradients'
assert all(m.last_audit['binary'] for m in model.modules() if isinstance(m, RateIF))
rate = RateIF(1, 4)
rate.audit = True
assert torch.equal(rate(torch.tensor([[[[0., .25, .5, 1.]]]])), torch.tensor([[[[0., .25, .5, 1.]]]]))
assert rate.last_audit['binary']
print('v2 smoke passed: shapes, binary IF, rate values, state reset, finite nonzero gradients')
