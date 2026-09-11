# Experiment 2: Independent Residual Refine

Experiment 2 keeps the trained ConvGuided model from `../exp1` frozen as
stage 1. It changes only refine: each of its three stages predicts a new
residual from the noisy image, its current estimate, and normalized Sobel
gradient. The estimates are accumulated as:

`coarse + delta1 -> stage1 + delta2 -> stage2 + delta3 -> final`

Training uses full BSDS400 images with rotations and horizontal/vertical
flips. BSD68 uses deterministic AWGN for validation. The training script saves
both `experiment2_sigma*_train_best.pth` and
`experiment2_sigma*_val_best.pth` under `models/sigma*/`.

```bash
cd /mnt/d/nuoqi/halo-free/Decoupled_Learning/20260621_denoiseretrain/exp2
CUDA_VISIBLE_DEVICES=1 python train_experiment2_fullimage.py --sigma 25 --device cuda:0
```

The original previously trained Experiment 2 checkpoint could not be restored;
new checkpoints will be written to this directory.
