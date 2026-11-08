#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TORCH_DDP_BACKEND="${TORCH_DDP_BACKEND:-nccl}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON="${PYTHON:-/home/nuoqi/miniconda3/envs/halofree/bin/python}"
OUTPUT="${OUTPUT:-expe/runs/scratch/exp7c_e0/seed42_exposure3840000}"
CACHE="${CACHE:-expc/runs/0818_memmap_4gpu/train_0818_float32.npy}"

resume_args=()
if [[ -f "$OUTPUT/latest.pth" ]]; then
  resume_args+=(--resume)
fi

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
  expc/src/train_expc.py --parallelism ddp --variant exp7c_e0 \
  --data-root /mnt/d/ResearchCode/S2R-HDR/data/20260818 \
  --cache-path "$CACHE" --output-dir "$OUTPUT" \
  --crop-size 128 --batch-size-per-gpu 16 --grad-accumulation 2 \
  --total-iter 60000 --val-count 200 --val-every 5000 --save-every 100 \
  --num-workers 8 --precision bf16 --lr 3e-4 --min-lr 1e-6 \
  --seed 42 --loss-mode base "${resume_args[@]}"
