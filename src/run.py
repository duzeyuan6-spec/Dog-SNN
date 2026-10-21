"""Pilot training/evaluation. Test split stays sealed until model selection ends."""
import argparse
import json
import os
import signal
import threading
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import lpips
import pynvml as nv
from common import config, write_json, seed_all, Frames, teacher
from model import StudentVAE

STOP = False


def stop_requested(*_):
    global STOP
    STOP = True


signal.signal(signal.SIGTERM, stop_requested)


def checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)


def student(cfg, kind):
    return StudentVAE(kind, cfg['width'], cfg['timesteps'])


def train(cfg, kind):
    root = Path(cfg['root'])
    dataset = Frames(cfg, 'train')
    model = student(cfg, kind).cuda().train()
    reference = teacher(cfg, 'cuda')
    perceptual = lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=1e-4)
    ckpt = root/'runs'/kind/'last.pt'
    start = 0
    if ckpt.exists():
        saved = torch.load(ckpt, map_location='cpu', weights_only=False)
        if saved['config'] != cfg:
            raise RuntimeError('Config changed: create a new run instead of silently resuming')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        start = saved['step']
        torch.set_rng_state(saved['cpu_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'])
    def save(step):
        checkpoint(ckpt, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                         'step': step, 'kind': kind, 'config': cfg,
                         'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state()})
    print(json.dumps({'event': 'training_started', 'kind': kind, 'step': start,
                      'parameters': sum(p.numel() for p in model.parameters())}), flush=True)
    for step in range(start, cfg['pilot_steps']):
        step_start = time.perf_counter()
        if STOP:
            save(step)
            raise SystemExit(75)
        # Stateless sample selection makes a resumed optimizer step use the same batch.
        rng = np.random.default_rng(cfg['seed'] + step)
        ids = rng.integers(0, len(dataset), size=cfg['batch_size'])
        x = torch.stack([dataset[int(i)][0] for i in ids]).cuda()
        with torch.no_grad():
            target = reference.encode(x).latent_dist.mode()
        optimizer.zero_grad(set_to_none=True)
        reconstruction, mu, logvar = model(x, sample=True)
        pixel = F.l1_loss(reconstruction, x)
        percept = perceptual(reconstruction, x).mean()
        latent = ((mu-target).square()/(target.square().mean().detach()+1e-4)).mean()
        kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).mean()
        loss = pixel + 0.25*percept + 0.05*latent + 1e-6*kl
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == start or (step+1) % 50 == 0:
            print(json.dumps({'step': step+1, 'loss': loss.item(), 'l1': pixel.item(),
                              'lpips': percept.item(), 'latent': latent.item(), 'kl': kl.item()}), flush=True)
        if step == start or (step+1) % 250 == 0:
            save(step+1)
        if os.environ.get('EXPERIMENT_SHARED_GPU') == '1':
            torch.cuda.synchronize()
            time.sleep((time.perf_counter()-step_start)*cfg['shared_training_policy']['pause_to_work_ratio'])
    save(cfg['pilot_steps'])


def ssim(x, y):
    # Standard local SSIM: RGB mean, Gaussian 11x11, sigma 1.5, valid support.
    g = torch.arange(11, device=x.device, dtype=x.dtype)-5
    g = (-g.square()/(2*1.5**2)).exp()
    g /= g.sum()
    kernel = (g[:, None]*g[None, :])[None, None].expand(3, 1, -1, -1)
    def filt(z):
        return F.conv2d(z, kernel, groups=3)
    mx, my = filt(x), filt(y)
    vx, vy, cov = filt(x*x)-mx*mx, filt(y*y)-my*my, filt(x*y)-mx*my
    value = ((2*mx*my+0.01**2)*(2*cov+0.03**2))/((mx*mx+my*my+0.01**2)*(vx+vy+0.03**2))
    return value.mean((1, 2, 3))


def exclusive(handle):
    from resource_queue import foreign_processes
    if foreign_processes(handle, {os.getpid()}):
        raise SystemExit(75)


def benchmark(fn, x, handle):
    exclusive(handle)
    for _ in range(20):
        fn(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    latencies = []
    wall_latencies = []
    for _ in range(150):
        if STOP:
            raise SystemExit(75)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter()
        begin.record()
        fn(x)
        end.record()
        torch.cuda.synchronize()
        wall_latencies.append((time.perf_counter()-wall)*1000)
        latencies.append(begin.elapsed_time(end))
    rounds = []
    for _ in range(3):
        exclusive(handle)
        samples = []
        sampling_stop = threading.Event()
        def power_sample():
            while not sampling_stop.is_set():
                try:
                    samples.append((time.perf_counter(), nv.nvmlDeviceGetPowerUsage(handle)/1000))
                except nv.NVMLError:
                    pass
                sampling_stop.wait(0.05)
        thread = threading.Thread(target=power_sample, daemon=True)
        thread.start()
        try:
            energy_start = nv.nvmlDeviceGetTotalEnergyConsumption(handle)
        except nv.NVMLError:
            energy_start = None
        t0 = time.perf_counter()
        count = 0
        while time.perf_counter()-t0 < 10:
            if STOP:
                sampling_stop.set()
                thread.join()
                raise SystemExit(75)
            for _ in range(16):
                fn(x)
            torch.cuda.synchronize()
            count += 16
        elapsed = time.perf_counter()-t0
        try:
            energy_end = nv.nvmlDeviceGetTotalEnergyConsumption(handle) if energy_start is not None else None
        except nv.NVMLError:
            energy_end = None
        sampling_stop.set()
        thread.join()
        exclusive(handle)
        method, energy = 'unavailable', None
        if energy_end is not None and energy_end > energy_start:
            method, energy = 'nvml_total_energy_mj', (energy_end-energy_start)/1000
        elif len(samples) >= 2:
            # Include endpoints by constant extension; retain raw samples for audit.
            ts, ps = zip(*samples)
            inside = [(max(t0, min(t0+elapsed, t)), p) for t, p in zip(ts, ps)]
            ts = [t0]+[t for t, _ in inside]+[t0+elapsed]
            ps = [ps[0]]+[p for _, p in inside]+[ps[-1]]
            energy = float(np.trapezoid(ps, ts))
            method = 'nvml_board_power_integral_50ms_polling'
        rounds.append({'images': count, 'seconds': elapsed, 'images_per_second': count/elapsed,
                       'gpu_joules_per_image': energy/count if energy is not None else None,
                       'energy_method': method, 'power_samples': samples})
    return {'scope': 'GPU-resident model forward; excludes file IO, CPU preprocessing and transfers',
            'precision': 'float32, TF32 disabled', 'batch_size': 1,
            'cuda_p50_ms': float(np.median(latencies)), 'cuda_p95_ms': float(np.percentile(latencies, 95)),
            'wall_p50_ms': float(np.median(wall_latencies)), 'wall_p95_ms': float(np.percentile(wall_latencies, 95)),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'rounds': rounds}


@torch.inference_mode()
def evaluate(cfg, kind, quality_only=False):
    root = Path(cfg['root'])
    nv.nvmlInit()
    handle = nv.nvmlDeviceGetHandleByUUID(os.environ['EXPERIMENT_NVML_UUID'])
    if kind == 'baseline':
        model = teacher(cfg, 'cuda')
        def fn(x):
            return model.decode(model.encode(x).latent_dist.mode()).sample
    else:
        model = student(cfg, kind).cuda().eval()
        saved = torch.load(root/'runs'/kind/'last.pt', map_location='cpu', weights_only=False)
        if saved['config'] != cfg or saved['step'] != cfg['pilot_steps']:
            raise RuntimeError('Checkpoint configuration/step mismatch')
        model.load_state_dict(saved['model'])
        def fn(x):
            return model(x)[0]
    metric = lpips.LPIPS(net='alex').cuda().eval()
    dataset = Frames(cfg, 'validation', cfg['validation_frames'])
    loader = DataLoader(dataset, batch_size=4, num_workers=0)
    rows = []
    for x, paths in loader:
        if STOP:
            raise SystemExit(75)
        x = x.cuda()
        raw = fn(x)
        if not torch.isfinite(raw).all():
            raise RuntimeError('Nonfinite reconstruction')
        y = raw.clamp(-1, 1)
        mse = ((x-y)/2).square().mean((1, 2, 3))
        psnr = -10*torch.log10(mse.clamp_min(1e-12))
        percept = metric(y, x).flatten()
        structural = ssim((x+1)/2, (y+1)/2)
        for i, path in enumerate(paths):
            rows.append({'path': path, 'psnr': psnr[i].item(), 'lpips': percept[i].item(), 'ssim': structural[i].item()})
        if len(rows) <= 4:
            from torchvision.utils import save_image
            sample_path = root/'runs'/kind/'reconstructions.png'
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            save_image(torch.cat([x, y]).cpu()/2+0.5, sample_path, nrow=len(paths))
    del metric
    torch.cuda.empty_cache()
    x = dataset[0][0].unsqueeze(0).cuda()
    perf = None if quality_only else benchmark(fn, x, handle)
    write_json(root/'runs'/kind/('quality.json' if quality_only else 'validation.json'), {
        'kind': kind, 'split': 'validation', 'pilot_only': True, 'frames': rows,
        'quality': {key: float(np.mean([r[key] for r in rows])) for key in ('psnr', 'ssim', 'lpips')},
        'performance': perf, 'shared_gpu': os.environ.get('EXPERIMENT_SHARED_GPU') == '1', 'gpu_uuid': os.environ['EXPERIMENT_NVML_UUID'],
        'gpu_name': torch.cuda.get_device_name(), 'parameters': sum(p.numel() for p in model.parameters()),
        'torch': torch.__version__, 'config': cfg})


def report(cfg):
    root = Path(cfg['root'])
    results = {kind: json.loads((root/'runs'/kind/'validation.json').read_text()) for kind in ('baseline', 'ann', 'snn')}
    b, s = results['baseline'], results['snn']
    quality_pass = (s['quality']['psnr'] >= b['quality']['psnr']-cfg['quality_gate']['psnr_drop_db'] and
                    s['quality']['lpips'] <= b['quality']['lpips']+cfg['quality_gate']['lpips_increase'])
    same_gpu = b['gpu_uuid'] == s['gpu_uuid']
    def energy(r):
        values = [x['gpu_joules_per_image'] for x in r['performance']['rounds']]
        return float(np.mean(values)) if all(x is not None for x in values) else None
    eb, es = energy(b), energy(s)
    time_pass = s['performance']['wall_p95_ms'] <= 0.9*b['performance']['wall_p95_ms']
    energy_pass = eb is not None and es is not None and es <= 0.9*eb
    write_json(root/'runs/pilot_report.json', {'status': 'pilot_candidate' if quality_pass and same_gpu and time_pass and energy_pass else 'pilot_gate_not_met',
        'quality_pass': quality_pass, 'same_physical_gpu': same_gpu, 'latency_pass': time_pass, 'energy_pass': energy_pass,
        'stage1_complete': False, 'stage2_authorized_by_results': False,
        'next_required': 'Review pilot; then freeze model and run paired, repeated held-out test before VLA integration.',
        'summary': {k: {'quality': r['quality'], 'wall_p95_ms': r['performance']['wall_p95_ms'],
                       'gpu_joules_per_image': energy(r)} for k, r in results.items()},
        'caveat': 'No robotics success results. Hybrid SNN; dense GPU kernels do not imply event-driven hardware energy savings.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['train', 'quality', 'evaluate', 'report'])
    parser.add_argument('kind', choices=['baseline', 'ann', 'snn', 'all'])
    args = parser.parse_args()
    cfg = config()
    seed_all(cfg['seed'])
    if os.environ.get('EXPERIMENT_SHARED_GPU') == '1':
        torch.cuda.set_per_process_memory_fraction(cfg['shared_training_policy']['memory_fraction'])
    {'train': lambda: train(cfg, args.kind), 'quality': lambda: evaluate(cfg, args.kind, True),
     'evaluate': lambda: evaluate(cfg, args.kind), 'report': lambda: report(cfg)}[args.action]()
