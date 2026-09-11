# DnCNN -> Refine bridge

This directory is isolated from `20260815_edgepreserving`. It freezes the shipped
PyTorch DnCNN sigma=25 model, generates full-image BSDS caches, then trains a
GroupNorm (`groupnorm`) or original BatchNorm (`original`) refine cascade.

```bash
cd /mnt/d/nuoqi/halo-free/Decoupled_Learning/20260621_denoiseretrain
PY=/home/nuoqi/miniconda3/envs/halofree/bin/python
ROOT=/mnt/d/nuoqi/halo-free/data_bsds
OUT=third_party/DnCNN/refine_bridge/runs/sigma25
$PY third_party/DnCNN/refine_bridge/generate_cache.py \
  --train-root "$ROOT/BSDS 400" --test-root "$ROOT/BSD68" \
  --out "$OUT/cache.pt" --device cpu
$PY third_party/DnCNN/refine_bridge/train_refine.py \
  --cache "$OUT/cache.pt" --variant groupnorm --epochs 80 \
  --out "$OUT/groupnorm" --device cpu
$PY third_party/DnCNN/refine_bridge/train_refine.py \
  --cache "$OUT/cache.pt" --variant original --epochs 80 \
  --out "$OUT/original" --device cpu
```

`metrics.json` reports the frozen DnCNN BSD68 baseline and refine PSNR.

To run all four experiments sequentially on GPUs 0-3:

```bash
bash third_party/run_refine_four_gpu_sequential.sh
```
