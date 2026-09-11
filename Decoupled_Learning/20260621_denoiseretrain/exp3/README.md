# Experiment 3: Deep Independent Residual Refine

Experiment 3 is Experiment 2 with only one architectural change: every refine
stage uses five `3x3` convolutional layers instead of three. It keeps
BatchNorm, max-normalized Sobel guidance, full-image BSDS400 training, geometry
augmentation, the frozen ConvGuided stage from `../exp1`, and BSD68 validation.

This experiment accepts sigma 25 only. It saves both train and validation best
checkpoints to `models/sigma25/` using the `experiment3_deep_` prefix.

```bash
cd /mnt/d/nuoqi/halo-free/Decoupled_Learning/20260621_denoiseretrain/exp3
CUDA_VISIBLE_DEVICES=1 python train_experiment3_fullimage.py --sigma 25 --device cuda:0
```
