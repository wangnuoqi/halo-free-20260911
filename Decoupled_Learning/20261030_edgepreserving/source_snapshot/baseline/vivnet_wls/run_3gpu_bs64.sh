#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

exec /home/nuoqi/miniconda3/envs/halofree/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=3 baseline/vivnet_wls/train.py \
  --data-root /mnt/d/ResearchCode/S2R-HDR/data/20260818 \
  --output-dir baseline/runs/vivnet_wls_patch256_nccl_3gpu_bs64 \
  --total-iter 100000 --batch-size-per-gpu 64 --micro-batch-size 16 \
  --train-crop-size 256 --precision bf16 --num-workers 4 "$@"
