#!/usr/bin/env bash
set -euo pipefail

# E2 paired architecture comparison:
#   legacy_matched_e0: old Exp7C LRNet
#   legacy_matched_e1: VIVNet U-Net
# Each variant runs base -> refine -> joint before the next variant starts.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TORCH_DDP_BACKEND="nccl"
export PYTHONUNBUFFERED="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_MODULE_LOADING="LAZY"
export NCCL_SHM_DISABLE="0"
export OMP_NUM_THREADS="1"

PYTHON="${PYTHON:-/home/nuoqi/miniconda3/envs/halofree/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/d/ResearchCode/S2R-HDR/data/20260818}"
RUN_ROOT="${RUN_ROOT:-expe/runs/legacy_matched_exposure3840000_v5}"
MASTER_PORT="${MASTER_PORT:-29562}"
TOTAL_ITER="${TOTAL_ITER:-60000}"
CROP_SIZE="${CROP_SIZE:-128}"
BATCH_PER_RANK="${BATCH_PER_RANK:-16}"
GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-42}"
if (( 2 * BATCH_PER_RANK * GRAD_ACCUMULATION != 64 || TOTAL_ITER != 60000 )); then
  echo "E1 Base exposure requires effective batch 64 and 60000 updates" >&2
  exit 2
fi

SCRIPT="expe/train_legacy_matched.py"

run_phase() {
  local variant="$1"
  local phase="$2"
  local output="$3"
  local base_checkpoint="${4:-}"
  local refine_checkpoint="${5:-}"

  if [[ -f "$output/final.pth" ]]; then
    echo "[skip] $variant $phase already has $output/final.pth"
    return
  fi

  local resume_args=()
  if [[ -f "$output/latest.pth" ]]; then
    resume_args+=(--resume)
  fi

  local init_args=()
  if [[ -n "$base_checkpoint" ]]; then
    init_args+=(--base-checkpoint "$base_checkpoint")
  fi
  if [[ -n "$refine_checkpoint" ]]; then
    init_args+=(--refine-checkpoint "$refine_checkpoint")
  fi

  echo "[start] $variant $phase"
  "$PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=2 \
    --master_port "$MASTER_PORT" \
    "$SCRIPT" \
    --parallelism ddp \
    --variant "$variant" \
    --phase "$phase" \
    --data-root "$DATA_ROOT" \
    --output-dir "$output" \
    "${init_args[@]}" \
    --crop-size "$CROP_SIZE" \
    --total-iter "$TOTAL_ITER" \
    --batch-size-per-rank "$BATCH_PER_RANK" \
    --grad-accumulation "$GRAD_ACCUMULATION" \
    --val-every 5000 \
    --save-every 1000 \
    --num-workers "$NUM_WORKERS" \
    --val-count 200 \
    --seed "$SEED" \
    --lr-base 1e-4 \
    --lr-refine 1e-2 \
    --lr-joint-coarse 3e-4 \
    --lr-joint-refine 3e-3 \
    --device cuda \
    "${resume_args[@]}"
}

run_variant() {
  local variant="$1"
  local root="$RUN_ROOT/${variant##legacy_matched_}"
  local base="$root/base"
  local refine="$root/refine"
  local joint="$root/joint"

  run_phase "$variant" base "$base"
  run_phase "$variant" refine "$refine" "$base/best.pth"
  run_phase "$variant" joint "$joint" "$base/best.pth" "$refine/best.pth"
}

run_variant legacy_matched_e0
run_variant legacy_matched_e1

echo "[done] E2 matched E0 and E1 base/refine/joint chains completed."
