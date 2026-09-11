#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="/mnt/d/nuoqi/halo-free/Decoupled-params/try-freezb"
CMD=(
  env
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
  /home/nuoqi/miniconda3/bin/conda run --no-capture-output -n halofree python -u
  "${ROOT_DIR}/eval_try_freezb.py"
  --variants v3 v4
  --model-kind "${MODEL_KIND:-best}"
  --gpu 0
  --batch-size "${BATCH_SIZE:-1}"
  --num-workers "${NUM_WORKERS:-0}"
)
if [[ "${SAVE_PREDS:-0}" == "1" ]]; then CMD+=(--save-preds); fi
"${CMD[@]}"
