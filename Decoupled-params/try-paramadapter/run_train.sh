#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/d/nuoqi/halo-free/Decoupled-params/try-paramadapter"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TRAIN_BATCH_PER_GPU="${TRAIN_BATCH_PER_GPU:-24}"
TEST_BATCH="${TEST_BATCH:-2}"
TRAIN_WORKERS="${TRAIN_WORKERS:-8}"
TEST_WORKERS="${TEST_WORKERS:-8}"
EPOCHS="${EPOCHS:-80}"
PIN_MEMORY="${PIN_MEMORY:-0}"
NO_RESUME="${NO_RESUME:-0}"

CMD=(
  env
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
  /home/nuoqi/miniconda3/bin/conda run --no-capture-output -n halofree python -u
  "${ROOT_DIR}/train_convguided_base_paramadapter_film.py"
  --save-dir "${ROOT_DIR}/model"
  --epochs "${EPOCHS}"
  --train-batch-per-gpu "${TRAIN_BATCH_PER_GPU}"
  --test-batch "${TEST_BATCH}"
  --train-workers "${TRAIN_WORKERS}"
  --test-workers "${TEST_WORKERS}"
)

if [[ "${PIN_MEMORY}" == "1" ]]; then
  CMD+=(--pin-memory)
fi

if [[ "${NO_RESUME}" == "1" ]]; then
  CMD+=(--no-resume)
fi

"${CMD[@]}"
