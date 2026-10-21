"""Validation-driven v2 training; no test data or shared-GPU energy claims."""
import json
import os
import signal
import time
from pathlib import Path
import numpy as np
import torch
import lpips
from torch.nn import functional as F
from torch.utils.data import DataLoader
from common import config, teacher, Frames, seed_all, write_json
from run import checkpoint, ssim
from spike_encoder import SpikeEncoder, RateIF

STOP = False
SPEC = dict(version='joint_t1_v6', width=128, depth=6, timesteps=1,
            steps=10000, warmup=0, transition=0, validation_every=1000,
            lr=0.00002, batch_size=4, latent_weight=0.05, logvar_weight=0.001,
            pixel_weight=1.0, lpips_weight=0.1, mse_weight=50.0,
            initialization='joint_t2_v5/best.pt', sample_offset=70000, decoder_lr_ratio=0.2)


def stop(*_):
    global STOP
    STOP = True


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)


def emit(event, **kw):
    print(json.dumps(dict(event=event, time=time.time(), **kw)), flush=True)


def pause(cfg, started):
    if os.environ.get('EXPERIMENT_SHARED_GPU') == '1':
        torch.cuda.synchronize()
        time.sleep((time.perf_counter()-started)*cfg['shared_training_policy']['pause_to_work_ratio'])


@torch.no_grad()
def validate(cfg, model, reference, perceptual, step, dest):
    model.eval()
    model.set_blend(1.0)
    modules = [m for m in model.modules() if isinstance(m, RateIF)]
    for m in modules:
        m.audit = True
    rows, audit = [], []
    loader = DataLoader(Frames(cfg, 'validation', cfg['validation_frames']), batch_size=4)
    for x, paths in loader:
        if STOP:
            raise SystemExit(75)
        started = time.perf_counter()
        x = x.cuda()
        mu, _ = model(x)
        y = reference.decode(mu).sample.clamp(-1, 1)
        if not torch.isfinite(y).all():
            raise RuntimeError('Nonfinite validation reconstruction')
        scores = {'psnr': -10*torch.log10(((x-y)/2).square().mean((1,2,3)).clamp_min(1e-12)),
                  'ssim': ssim((x+1)/2, (y+1)/2), 'lpips': perceptual(y, x).flatten()}
        rows.extend(dict(path=p, **{k:v[i].item() for k,v in scores.items()}) for i,p in enumerate(paths))
        if not audit:
            audit = [m.last_audit for m in modules]
            for m in modules:
                m.audit = False
        pause(cfg, started)
    if not all(a['binary'] for a in audit):
        raise RuntimeError('Nonbinary spike detected')
    quality = {k: float(np.mean([r[k] for r in rows])) for k in ('psnr','ssim','lpips')}
    baseline = json.loads((Path(cfg['root'])/'runs/baseline/quality.json').read_text())['quality']
    passed = (quality['psnr'] >= baseline['psnr']-cfg['quality_gate']['psnr_drop_db'] and
              quality['lpips'] <= baseline['lpips']+cfg['quality_gate']['lpips_increase'])
    result = dict(step=step, quality=quality, baseline=baseline, quality_pass=passed,
                  split='validation', frames=rows, performance=None, stage1_complete=False,
                  spike_audit=audit, blend=1.0, spec=SPEC,
                  architecture='spiking residual encoder with analog shortcuts; pretrained decoder jointly fine-tuned on train split')
    write_json(dest/f'validation_{step:06d}.json', result)
    write_json(dest/'quality.json', result)
    emit('validation', step=step, quality=quality, quality_pass=passed, blend=1.0)
    model.train()
    return result


def main():
    cfg = config()
    seed_all(cfg['seed'])
    if os.environ.get('EXPERIMENT_SHARED_GPU') == '1':
        torch.cuda.set_per_process_memory_fraction(cfg['shared_training_policy']['memory_fraction'])
    dest = Path(cfg['root'])/'runs'/SPEC['version']
    dest.mkdir(parents=True, exist_ok=True)
    write_json(dest/'spec.json', SPEC)
    model = SpikeEncoder(SPEC['width'], SPEC['depth'], SPEC['timesteps']).cuda().train()
    reference = teacher(cfg, 'cuda')
    perceptual = lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False)
    reference.decoder.requires_grad_(True)
    parameters = list(model.parameters()) + list(reference.decoder.parameters())
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr_ratio': 1.0},
        {'params': reference.decoder.parameters(), 'lr_ratio': SPEC['decoder_lr_ratio']}
    ], lr=SPEC['lr'], weight_decay=1e-4)
    dataset = Frames(cfg, 'train')
    start, best = 0, -float('inf')
    latest = dest/'last.pt'
    if latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        if saved['spec'] != SPEC or saved['config'] != cfg:
            raise RuntimeError('Changed run specification; create a new version')
        model.load_state_dict(saved['model'])
        reference.decoder.load_state_dict(saved['decoder'])
        optimizer.load_state_dict(saved['optimizer'])
        start, best = saved['step'], saved['best_psnr']
        torch.set_rng_state(saved['cpu_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'])
    else:
        source = Path(cfg['root'])/'runs'/SPEC['initialization']
        saved = torch.load(source, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'])
        reference.decoder.load_state_dict(saved['decoder'])
        import hashlib
        write_json(dest/'initialization.json', dict(source=str(source), step=saved['step'],
                   sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        emit('initialized_from_previous_best', source=str(source), source_step=saved['step'])
    def save(step, path=latest):
        checkpoint(path, dict(model=model.state_dict(), decoder=reference.decoder.state_dict(), optimizer=optimizer.state_dict(),
                             step=step, best_psnr=best, spec=SPEC, config=cfg,
                             cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()))
    emit('training_started', step=start, total_steps=SPEC['steps'], spec=SPEC,
         encoder_parameters=sum(p.numel() for p in model.parameters()))
    if start == 0 and not (dest/'validation_000000.json').exists():
        result = validate(cfg, model, reference, perceptual, 0, dest)
        best = result['quality']['psnr']
        save(0, dest/'best.pt')
        write_json(dest/'best_quality.json', result)
        save(0)
    for step in range(start, SPEC['steps']):
        if STOP:
            save(step)
            raise SystemExit(75)
        started = time.perf_counter()
        blend = 1.0 if SPEC['transition'] == 0 else min(1.0, max(0.0, (step-SPEC['warmup'])/SPEC['transition']))
        model.set_blend(blend)
        lr = SPEC['lr'] * (0.1+0.9*0.5*(1+np.cos(np.pi*step/SPEC['steps'])))
        for group in optimizer.param_groups:
            group['lr'] = lr * group['lr_ratio']
        ids = np.random.default_rng(cfg['seed']+SPEC['sample_offset']+step).integers(len(dataset), size=SPEC['batch_size'])
        x = torch.stack([dataset[int(i)][0] for i in ids]).cuda()
        with torch.no_grad():
            target = reference.encode(x).latent_dist
            target_mu, target_logvar = target.mean, target.logvar
            scale = target_mu.square().mean((0,2,3), keepdim=True).clamp_min(0.01)
        optimizer.zero_grad(set_to_none=True)
        mu, logvar = model(x)
        reconstruction = reference.decode(mu).sample
        latent = ((mu-target_mu).square()/scale).mean()
        variance = F.smooth_l1_loss(logvar, target_logvar)
        pixel = F.l1_loss(reconstruction, x)
        mse = ((reconstruction-x)/2).square().mean()
        percept = perceptual(reconstruction, x).mean()
        loss = (SPEC['latent_weight']*latent + SPEC['logvar_weight']*variance +
                SPEC['pixel_weight']*pixel + SPEC['lpips_weight']*percept + SPEC['mse_weight']*mse)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        if step == start:
            assert not any(p.requires_grad for p in reference.encoder.parameters())
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in reference.decoder.parameters())
            emit('gradient_audit', encoder_teacher_frozen=True, decoder_receives_gradient=True)
        optimizer.step()
        if step == start or (step+1)%50 == 0:
            progress = dict(step=step+1, total_steps=SPEC['steps'], blend=blend, lr=lr,
                            loss=loss.item(), latent=latent.item(), l1=pixel.item(), lpips=percept.item(), mse=mse.item())
            emit('train', **progress)
            write_json(dest/'progress.json', dict(updated_at=time.time(), **progress))
        if step == start or (step+1)%250 == 0:
            save(step+1)
        pause(cfg, started)
        if (step+1)%SPEC['validation_every'] == 0:
            save(step+1)
            result = validate(cfg, model, reference, perceptual, step+1, dest)
            if result['quality']['psnr'] > best:
                best = result['quality']['psnr']
                save(step+1, dest/'best.pt')
                write_json(dest/'best_quality.json', result)
            save(step+1)
    # Resume at final step must also recover a validation interrupted by resource yield.
    final_path = dest/f"validation_{SPEC['steps']:06d}.json"
    if not final_path.exists():
        result = validate(cfg, model, reference, perceptual, SPEC['steps'], dest)
        if result['quality']['psnr'] > best:
            best = result['quality']['psnr']
            save(SPEC['steps'], dest/'best.pt')
            write_json(dest/'best_quality.json', result)
    save(SPEC['steps'])
    emit('candidate_training_complete', next_action='Review best validation; keep test sealed; no success claim')


if __name__ == '__main__':
    main()
