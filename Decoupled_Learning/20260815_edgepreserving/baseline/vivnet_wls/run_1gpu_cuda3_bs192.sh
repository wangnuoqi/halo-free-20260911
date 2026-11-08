#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
unset CUDA_VISIBLE_DEVICES WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

exec /home/nuoqi/miniconda3/envs/halofree/bin/python -u baseline/vivnet_wls/train.py \
  --data-root /mnt/d/ResearchCode/S2R-HDR/data/20260818 \
  --output-dir baseline/runs/vivnet_wls_patch256_1gpu_cuda3_bs192 \
  --gpu-id 3 \
  --total-iter 100000 --batch-size-per-gpu 192 --micro-batch-size 16 \
  --train-crop-size 256 --precision bf16 --num-workers 4 "$@"
