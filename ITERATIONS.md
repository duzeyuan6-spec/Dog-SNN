# SNN–VAE iteration log

## Pilot v1 — quality failed

Fixed 400-frame validation subset, episode-disjoint train/validation/test splits:

| Model | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|
| Original FLUX.2 VAE | 38.08098 | 0.98312 | 0.00534 |
| Compact ANN student, 5,000 steps | 31.30176 | 0.93963 | 0.02697 |
| Compact hybrid SNN, 5,000 steps | 28.08591 | 0.88890 | 0.05353 |

Neither student passes the original quality gates. No energy or latency
conclusion is available. Keep all pilot checkpoints and results. The queue
advances to another candidate instead of waiting to benchmark this failed one.

## Encoder v2 — training candidate

Replace the encoder only; keep the original pretrained FLUX.2 decoder frozen.
Pixel-unshuffle preserves input samples, followed by width-128 residual
processing at 64×64 and 32×32. IF branches generate binary spikes over four
steps. Continuous residual shortcuts, latent moments, and decoder remain;
this is a hybrid SpikeVAE, not a fully spiking network.

The IF rate is averaged before the following linear convolution. GPU execution
uses dense kernels. Actual energy savings are unknown and must be measured.
Check that the pulse branches materially affect reconstruction before accepting
this architecture; binary spike existence alone is insufficient evidence.

Training: 2,000 continuous warmup steps, 4,000 transition steps, then full IF
mode through step 20,000. Distill teacher posterior means and log variances;
also optimize image L1 and LPIPS through the frozen decoder. Reconstruction
evaluation uses posterior means, as does the original baseline. Stochastic
posterior behavior remains a separate check before any broader VAE claims.

Every 1,000 steps, evaluate all 400 fixed validation images in full IF mode,
including during warmup. Save per-image scores, pulse audit, latest and best
checkpoints. Selection uses validation only. Test data remain sealed.

Remote progress:

```sh
tail -n 10 -F /datasets2/dzy/snn_vae_20261015/logs/encoder_v2.log
```

Completion of this candidate triggers review and further iteration, not a
success declaration. A quality-qualified candidate still requires matched ANN
controls, frozen configuration, independent test, repeated seeds and paired
exclusive-GPU latency/energy tests. Stage 1 requires existing quality gates
and at least 10% improvement in both runtime and measured board energy. Only
then proceed to the VLA task evaluation.

## Encoder v2 completed; encoder v3 launched

V2 best: step 19,000, PSNR 33.93805 dB, SSIM 0.96426, LPIPS 0.02155.
Final step 20,000: PSNR 33.93598 dB. The PSNR threshold remains 37.58098 dB;
v2 failed. Binary IF audit passed, with branch firing rates around 0.24–0.28
on the audited batch. No energy or speed result is available.

V3 keeps the architecture and frozen decoder, initializes from v2 best.pt,
and trains 20,000 additional steps in full spike mode. Reduce latent loss
weight from 1.0 to 0.05, add 50× image MSE on the [0,1] scale, use L1 weight
1.0 and LPIPS weight 0.1. Learning rate starts at 5e-5 and decays to 5e-6.
Use a new training batch seed offset; preserve original validation images.
This tests whether the latent-dominated objective was limiting reconstruction;
improvement is a hypothesis, not an established result.

V2 source archived on server under revisions/encoder_v2/src; v3 logs are
logs/encoder_v3.log and outputs runs/encoder_v3. Initialization records the
parent checkpoint path, step and SHA-256. Test data remain sealed.

## Encoder v3 completed; joint v4

V3 best/final at step 20,000: PSNR 35.44796 dB, SSIM 0.97209,
LPIPS 0.01605. Quality still fails the unchanged PSNR gate. Late gains
were small (35.40117 at step 16,000).

V4 initializes the encoder from v3 best.pt and jointly fine-tunes the
pretrained decoder with a learning rate one fifth of the encoder rate.
Other v3 loss weights and full binary IF mode remain. The teacher encoder
stays frozen; original baseline model files remain unchanged. Checkpoints
now include decoder parameters as well as encoder and optimizer states.
This tests decoder adaptation to the student's latent errors. It does not
establish efficiency benefits or isolate spiking from fine-tuning benefits;
matched ANN and fine-tuned baseline controls are still required.

Remote logs: logs/joint_v4.log; results: runs/joint_v4. Plan: 20,000 steps,
validation every 1,000 steps. Test data remain sealed.

## Full validation and pulse audit completed

Full validation (3,018 frames): baseline PSNR 38.06114 / LPIPS 0.005348;
hybrid SNN PSNR 39.43995 / LPIPS 0.004987. Quality gate passed.
On the same 400-image subset, SNN PSNR 39.41934, zero-pulse ablation
21.40190, continuous activation with unchanged weights 36.59128.
Pulse branches contribute, but this does not establish superiority over a
trained ANN. Test data and energy measurements remain untouched.

## Matched ANN control queued

ann_stage2.py → ann_stage3.py → ann_stage4.py reproduce the three successful
SNN training phases (20,000 steps each), using their data seeds, losses,
learning rate schedules and decoder freeze/fine-tune choices. AnalogEncoder
replaces each RateIF with the continuous clipped activation and removes
spike computation entirely. Parameter shapes are identical. Control starts
from scratch, and each subsequent phase uses its own best checkpoint.
Parameter equality, agreement with the continuous activation limit, absence
of spike modules and finite gradients were checked on the server.

The control tests this specific matched recipe; stronger ANN tuning may be
needed before making claims about SNN-specific benefits. Fine-tuned original
VAE controls and independent multi-seed evaluation remain outstanding.
