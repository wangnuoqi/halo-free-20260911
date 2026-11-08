#!/usr/bin/env bash
set -euo pipefail

# Archived original fixed-statistics A1/A2 launcher; retained for provenance.
# Both variants use the VIVNet LR predictor and the current 128/64 patch
# protocol; no external three-stage Refine/Joint is attached here.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export TORCH_DDP_BACKEND="nccl"
export PYTHONUNBUFFERED="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_MODULE_LOADING="LAZY"
export NCCL_SHM_DISABLE="0"
export OMP_NUM_THREADS="1"

PYTHON="${PYTHON:-/home/nuoqi/miniconda3/envs/halofree/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/d/ResearchCode/S2R-HDR/data/20260818}"
CACHE_PATH="${CACHE_PATH:-expc/runs/0818_memmap_4gpu/train_0818_float32.npy}"
RUN_ROOT="${RUN_ROOT:-expe/runs/a_series}"
MASTER_PORT="${MASTER_PORT:-29563}"

run_variant() {
  local variant="$1"
  local output="$RUN_ROOT/$variant"

  if [[ -f "$output/final.pth" ]]; then
    echo "[skip] $variant already has $output/final.pth"
    return
  fi

  local resume_args=()
  if [[ -f "$output/latest.pth" ]]; then
    resume_args+=(--resume)
  fi

  echo "[start] $variant"
  "$PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=2 \
    --master_port "$MASTER_PORT" \
    expc/src/train_expc.py \
    --parallelism ddp \
    --variant "$variant" \
    --data-root "$DATA_ROOT" \
    --cache-path "$CACHE_PATH" \
    --output-dir "$output" \
    --crop-size 128 \
    --batch-size-per-gpu 16 \
    --total-iter 60000 \
    --val-count 200 \
    --val-every 5000 \
    --save-every 100 \
    --num-workers 4 \
    --precision bf16 \
    --lr 3e-4 \
    --min-lr 1e-6 \
    --seed 42 \
    --loss-mode base \
    "${resume_args[@]}"
}

run_variant viv_gf_a1
run_variant viv_gf_a2

echo "[done] A1 and A2 coarse-only training completed."
