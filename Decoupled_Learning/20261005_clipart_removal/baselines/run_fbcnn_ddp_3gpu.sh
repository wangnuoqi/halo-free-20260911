#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export PYTHONUNBUFFERED=1
export USE_LIBUV=0

python -u train_fbcnn_clipart_ddp.py \
  --resume "$PWD/ckpt/fbcnn_rgb_q30_protocol160/fbcnn_last.pth" \
  --output-dir "$PWD/ckpt/fbcnn_rgb_q30_protocol160" \
  --epochs 160 \
  --batch-size 8 \
  --patch-size 128 \
  --lr 2e-5 \
  --val-every 5 \
  --num-workers 16 \
  --seed 42 \
  "$@"
