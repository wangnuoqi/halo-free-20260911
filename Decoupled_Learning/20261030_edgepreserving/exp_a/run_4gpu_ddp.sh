#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/nuoqi/miniconda3/envs/halofree/bin/python}"
VARIANT="${VARIANT:-a1}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/exp_a/runs/${VARIANT}_ddp4_patch128}"
EXTRA_ARGS=()
if [[ "${INIT_EXP7C:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--init-exp7c --finetune-lr "${FINETUNE_LR:-3e-5}")
fi

cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_MODULE_LOADING=LAZY \
NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}" \
OMP_NUM_THREADS=1 \
"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  --master_port="${MASTER_PORT:-29561}" \
  exp_a/src/train_exp_a.py \
  --variant "${VARIANT}" \
  --data-root /mnt/d/ResearchCode/S2R-HDR/data/20260818 \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size-per-gpu "${BATCH_SIZE_PER_GPU:-4}" \
  --crop-size "${CROP_SIZE:-128}" \
  --epochs "${EPOCHS:-200}" \
  --lr "${LR:-3e-4}" \
  --min-lr "${MIN_LR:-3e-6}" \
  --lr-schedule "${LR_SCHEDULE:-cosine}" \
  --precision "${PRECISION:-bf16}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --max-steps "${MAX_STEPS:-0}" \
  "${EXTRA_ARGS[@]}"
