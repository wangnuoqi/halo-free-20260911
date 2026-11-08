#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

exec /home/nuoqi/miniconda3/envs/halofree/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 baseline/vivnet_wls/train.py \
  --data-root /mnt/d/ResearchCode/S2R-HDR/data/20260818 \
  --output-dir baseline/runs/vivnet_wls_patch256_1gpu_cuda3_bs192 \
  --total-iter 100000 --batch-size-per-gpu 48 --micro-batch-size 16 \
  --train-crop-size 256 --precision bf16 --num-workers 4 "$@"
