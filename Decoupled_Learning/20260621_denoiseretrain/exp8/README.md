# Exp8: Denoiser-Specific Patch Protocols

Exp8 keeps each denoiser's patch geometry while making the experiment
self-contained under `exp8/`. `run_exp8.py` stages DnCNN/FFDNet initializers,
trains a fresh ConvGuided `128 HR -> 64 LR` base, and produces frozen/joint
bundles for A/B/C under `exp8/runs/`.

- A: ConvGuided, 128 HR patches, 32 patches/image, global batch 32, 80 epochs.
- B: DnCNN, 40x40 dense grid with stride 10 and four bicubic scales, batch 128,
  50 epochs; the joint denoiser schedule is 1e-3 for 30 epochs then 1e-4.
- C: FFDNet, 70x70 dense grid with stride 60, one random 0.45--1.45 scale per
  image, regenerated every 10 epochs, batch 64, 500 epochs.

Frozen runs use Exp6D; joint runs use Exp7C RF and coarse fidelity loss.
