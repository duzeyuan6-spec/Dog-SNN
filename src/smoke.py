"""CPU-only numerical checks; these do not measure or claim performance."""
import json
import torch
from common import config, seed_all, write_json
from model import StudentVAE, TemporalActivation
from pathlib import Path


def main():
    cfg = config()
    seed_all(cfg['seed'])
    summary = {}
    for kind in ('ann', 'snn'):
        model = StudentVAE(kind, width=16, timesteps=2)
        x = torch.rand(2, 3, 32, 32)*2-1
        y, mu, lv = model(x, sample=True)
        assert y.shape == x.shape and mu.shape == (2, 32, 4, 4)
        loss = (y-x).square().mean()+1e-6*(mu.square()+lv.exp()-lv-1).mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert model.encoder[0].conv.weight.grad.abs().sum() > 0
        with torch.no_grad():
            first = model(x)[0]
            model(torch.zeros_like(x))
            second = model(x)[0]
            assert torch.equal(first, second), 'Temporal state leaked between observations'
        summary[kind] = {'finite_gradients': True, 'reset_state': True,
                         'parameters': sum(p.numel() for p in model.parameters())}
    assert summary['ann']['parameters'] == summary['snn']['parameters']
    activation = TemporalActivation('snn')
    spikes = activation(torch.randn(4, 2, 3, 4, 4))
    assert ((spikes == 0) | (spikes == 1)).all()
    from diffusers import AutoencoderKLFlux2
    from run import ssim
    x = torch.rand(2, 3, 32, 32)
    assert torch.allclose(ssim(x, x), torch.ones(2), atol=1e-5)
    summary['binary_spikes'] = True
    summary['metric_identity'] = True
    summary['flux2_class_import'] = AutoencoderKLFlux2.__name__
    write_json(Path(cfg['root'])/'smoke_cpu.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
