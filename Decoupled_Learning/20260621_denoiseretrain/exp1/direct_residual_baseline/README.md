# Experiment A — Direct Residual Baseline

This isolated experiment tests whether the ConvGuided coarse image and the
coarse-to-refine interface limit the current result. It uses the same three-stage
refine parameterization, Sobel max normalization, BatchNorm, edge-guided loss,
whole-image BSDS400 training, BSD68 per-epoch validation, 80 epochs, and
geometry-only augmentation. The only architectural change is that the model
predicts `clean - noisy` directly and never receives a ConvGuided coarse image.

All outputs are written under this experiment directory: `models/`, `log_dir/`,
and `eval_results/`. The default output root is derived from the script location,
so moving this directory keeps its outputs with it.

Run one sigma per GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python train_direct_residual_fullimage.py --sigma 15 --device cuda:0
CUDA_VISIBLE_DEVICES=1 python train_direct_residual_fullimage.py --sigma 25 --device cuda:0
CUDA_VISIBLE_DEVICES=2 python train_direct_residual_fullimage.py --sigma 50 --device cuda:0
```

Each sigma saves both the lowest-training-MSE and lowest-BSD68-MSE checkpoints:

```text
models/sigma25/direct_residual_fullimage_sigma25_train_best.pth
models/sigma25/direct_residual_fullimage_sigma25_val_best.pth
```

Evaluate the BSD68-selected checkpoint and write predicted images plus metrics:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_direct_residual_fullimage.py --sigma 25 --device cuda:0
```
