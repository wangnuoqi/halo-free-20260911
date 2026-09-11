#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="/mnt/d/nuoqi/halo-free/Decoupled-params/try-freezb"
CMD=(
  env
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
  /home/nuoqi/miniconda3/bin/conda run --no-capture-output -n halofree python -u
  "${ROOT_DIR}/train_convguided_base_paraminput_freezeb_v2.py"
  --train-batch-per-gpu "${TRAIN_BATCH_PER_GPU:-12}"
  --test-batch "${TEST_BATCH:-4}"
  --train-workers "${TRAIN_WORKERS:-12}"
  --test-workers "${TEST_WORKERS:-2}"
  --epochs "${EPOCHS:-80}"
)
if [[ "${PIN_MEMORY:-0}" == "1" ]]; then CMD+=(--pin-memory); fi
if [[ "${NO_RESUME:-0}" == "1" ]]; then CMD+=(--no-resume); fi
"${CMD[@]}"
