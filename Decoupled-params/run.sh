#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/d/nuoqi/halo-free/Decoupled-params"
MODEL_KIND="${1:-best}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SAVE_PREDS="${SAVE_PREDS:-0}"

CMD=(
  /home/nuoqi/miniconda3/bin/conda run --no-capture-output -n halofree python -u
  "${ROOT_DIR}/eval_frozen_models.py"
  --gpu "${GPU_ID}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --model-kind "${MODEL_KIND}"
)

if [[ "${SAVE_PREDS}" == "1" ]]; then
  CMD+=(--save-preds)
fi

"${CMD[@]}"
