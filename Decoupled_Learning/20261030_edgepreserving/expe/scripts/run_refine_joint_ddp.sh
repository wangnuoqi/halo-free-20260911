#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
phase="${1:?Usage: bash expe/scripts/run_refine_joint_ddp.sh refine|joint|all [extra training arguments]}"
shift
case "$phase" in refine|joint|all) ;; *) echo 'phase must be refine, joint, or all' >&2; exit 2 ;; esac
variant="${VARIANT:-viv_gf_e1}"
base_run=seed42
if [[ "$variant" == viv_gf_e1 ]]; then base_run=seed42_ddpfix; fi
if [[ "$variant" == exp7c_e0 ]]; then base_run=seed42_exposure3840000; fi
base_checkpoint="${BASE_CHECKPOINT:-expe/runs/scratch/$variant/$base_run/best.pth}"
downstream_run=seed42
if [[ "$variant" == exp7c_e0 ]]; then downstream_run=seed42_exposure3840000; fi
refine_dir="${REFINE_DIR:-expe/runs/three_stage/$variant/$downstream_run/refine}"
joint_dir="${JOINT_DIR:-expe/runs/three_stage/$variant/$downstream_run/joint}"
if [[ ! -f "$base_checkpoint" ]]; then
  echo "Base checkpoint missing: $base_checkpoint. Set BASE_CHECKPOINT to the completed Base best.pth." >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TORCH_DDP_BACKEND=nccl PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_MODULE_LOADING=LAZY
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"

run_phase() {
  local current_phase="$1"
  shift
  local output learning_rate refine_checkpoint
  local phase_extra=()
  if [[ "$current_phase" == refine ]]; then
    output="$refine_dir"
    learning_rate="${REFINE_LR:-3e-4}"
  else
    output="$joint_dir"
    learning_rate="${JOINT_LR:-1e-5}"
    refine_checkpoint="${REFINE_CHECKPOINT:-$refine_dir/refine_best.pth}"
    if [[ ! -f "$refine_checkpoint" ]]; then
      echo "Refine checkpoint missing: $refine_checkpoint" >&2
      exit 2
    fi
    phase_extra+=(--refine-checkpoint "$refine_checkpoint")
  fi
  mkdir -p "$output"
  echo "Starting $current_phase for $variant; output=$output"
  "${TRAIN_PYTHON:-/home/nuoqi/miniconda3/envs/halofree/bin/python}" -m torch.distributed.run \
    --standalone --nproc_per_node="${NPROC_PER_NODE:-4}" --module exp_b.src.train_exp_b \
    --parallelism ddp --expe-variant "$variant" --phase "$current_phase" \
    --data-root "${DATA_ROOT:-/mnt/d/ResearchCode/S2R-HDR/data/20260818}" \
    --base-checkpoint "$base_checkpoint" --output-dir "$output" \
    --crop-size 128 --batch-size-per-rank "${BATCH_PER_RANK:-16}" --epochs "${EPOCHS:-200}" \
    --num-workers "${NUM_WORKERS:-8}" --precision bf16 --seed 42 --val-count 200 \
    --full-image-validation --lr "$learning_rate" --min-lr 1e-6 \
    "${phase_extra[@]}" "$@" 2>&1 | tee -a "$output/console.log"
}

case "$phase" in
  refine|joint)
    run_phase "$phase"
    ;;
  all)
    run_phase refine
    echo "Refine completed successfully; starting joint training."
    run_phase joint
    ;;
esac
