#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/nuoqi/miniconda3/envs/halofree/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
CROP_SIZE="${CROP_SIZE:-128}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-8}"
EPOCHS="${EPOCHS:-80}"
LR="${LR:-3e-4}"
PRECISION="${PRECISION:-bf16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-0}"
MASTER_PORT="${MASTER_PORT:-29561}"
NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/runs/sequence_a1_c1_a0_$(date +%Y%m%d_%H%M%S)}"

IFS=',' read -r -a GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#GPU_LIST[@]}" -ne 4 ]]; then
  echo "Expected exactly four visible GPU IDs, got '${CUDA_VISIBLE_DEVICES}'" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}/logs"
cd "${PROJECT_ROOT}"

echo "Sequential Experiment A training"
echo "GPUs=${CUDA_VISIBLE_DEVICES}; crop=${CROP_SIZE}; batch/GPU=${BATCH_SIZE_PER_GPU}; global batch=$((BATCH_SIZE_PER_GPU * 4))"
echo "epochs=${EPOCHS}; lr=${LR}; precision=${PRECISION}; max_steps=${MAX_STEPS}"
echo "run root=${RUN_ROOT}"
echo "Each variant starts from scratch; a failure stops the sequence."

for variant in a1 c1 a0; do
  output_dir="${RUN_ROOT}/${variant}"
  log_file="${RUN_ROOT}/logs/${variant}.log"

  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Starting ${variant}"
  env \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    VARIANT="${variant}" \
    OUTPUT_DIR="${output_dir}" \
    CROP_SIZE="${CROP_SIZE}" \
    BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU}" \
    EPOCHS="${EPOCHS}" \
    LR="${LR}" \
    PRECISION="${PRECISION}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    MAX_STEPS="${MAX_STEPS}" \
    MASTER_PORT="${MASTER_PORT}" \
    NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE}" \
    bash exp_a/run_4gpu_ddp.sh 2>&1 | tee "${log_file}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Finished ${variant}"
done

echo
echo "All three runs completed. Checkpoints and configs are under ${RUN_ROOT}/{a1,c1,a0}; logs are under ${RUN_ROOT}/logs/."
