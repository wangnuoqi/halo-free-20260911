#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DDP_BACKEND=nccl
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=1

PYTHON="${PYTHON:-/home/nuoqi/miniconda3/envs/halofree/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/d/ResearchCode/S2R-HDR/data/20260818}"
RUN_ROOT=expe/runs/legacy_matched_exposure3840000_v5

if [[ "${1:-}" == --dry-run ]]; then
  echo "E0 new Base/Refine/Joint -> E0 old Base/Refine/Joint -> E1 old Base -> verify/archive"
  echo "DDP/NCCL: CUDA_VISIBLE_DEVICES=0,1; 8 workers/rank"
  exit 0
fi

"$PYTHON" -c 'import torch; assert torch.cuda.device_count() == 2, "CUDA preflight requires exactly two visible GPUs"; print("CUDA preflight: 2 visible GPUs", flush=True)'
"$PYTHON" -m expe.verify_archive_e_2x2 --baseline-only

e0_output=expe/runs/scratch/exp7c_e0/seed42_exposure3840000
if [[ ! -f "$e0_output/final.pth" ]]; then
  echo "[1/3] E0 + new recipe"
  bash expe/scripts/run_e0_new_matched_ddp_2gpu.sh
else
  echo "[1/3] E0 + new recipe: final checkpoint exists; verifying after all stages"
fi

run_new_downstream() {
  local phase="$1"
  local output=expe/runs/three_stage/exp7c_e0/seed42_exposure3840000/$phase
  if [[ -f "$output/${phase}_final.pth" ]]; then
    echo "[skip] E0 new $phase final checkpoint exists"
    return
  fi
  local resume_args=()
  if [[ -f "$output/${phase}_latest.pth" ]]; then
    resume_args+=(--resume)
  fi
  VARIANT=exp7c_e0 TRAIN_PYTHON="$PYTHON" DATA_ROOT="$DATA_ROOT" \
    NPROC_PER_NODE=2 BATCH_PER_RANK=32 \
    NUM_WORKERS=8 EPOCHS=200 \
    bash expe/scripts/run_refine_joint_ddp.sh "$phase" "${resume_args[@]}"
}

run_new_downstream refine
run_new_downstream joint

run_old_phase() {
  local variant="$1"
  local slot="$2"
  local phase="$3"
  local output="$RUN_ROOT/$slot/$phase"
  if [[ -f "$output/final.pth" ]]; then
    echo "[skip] $variant $phase final checkpoint exists"
    return
  fi
  local resume_args=()
  if [[ -f "$output/latest.pth" ]]; then
    resume_args+=(--resume)
  fi
  local init_args=()
  if [[ "$phase" == refine || "$phase" == joint ]]; then
    init_args+=(--base-checkpoint "$RUN_ROOT/$slot/base/best.pth")
  fi
  if [[ "$phase" == joint ]]; then
    init_args+=(--refine-checkpoint "$RUN_ROOT/$slot/refine/best.pth")
  fi
  mkdir -p "$output"
  local log_file="$output/console.log"
  echo "[start] $variant $phase; log=$log_file"
  set +e
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
    expe/train_legacy_matched.py --parallelism ddp --variant "$variant" --phase "$phase" \
    --data-root "$DATA_ROOT" --output-dir "$output" \
    "${init_args[@]}" \
    --crop-size 128 --batch-size-per-rank 16 --grad-accumulation 2 \
    --total-iter 60000 --val-count 200 --val-every 5000 --save-every 1000 \
    --num-workers 8 --seed 42 --lr-base 1e-4 --lr-refine 1e-2 \
    --lr-joint-coarse 3e-4 --lr-joint-refine 3e-3 --device cuda \
    "${resume_args[@]}" 2>&1 | tee -a "$log_file"
  local status=${PIPESTATUS[0]}
  set -e
  if (( status != 0 )); then
    echo "[failed] $variant $phase (exit $status); checkpoint/log retained at $output" >&2
    return "$status"
  fi
  echo "[done] $variant $phase"
}

echo "[2/3] E0 + old recipe"
run_old_phase legacy_matched_e0 e0 base
run_old_phase legacy_matched_e0 e0 refine
run_old_phase legacy_matched_e0 e0 joint
echo "[3/3] E1 + old recipe"
run_old_phase legacy_matched_e1 e1 base

"$PYTHON" -m expe.verify_archive_e_2x2 --archive
echo "[done] Three matched cells finished; old E checkpoints archived."
