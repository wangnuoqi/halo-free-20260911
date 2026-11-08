#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-halofree}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0}"
CONV_SCRIPT="${SCRIPT_DIR}/train_convguided_20260818_l0p500_a1p400.py"
RF_SCRIPT="${SCRIPT_DIR}/train_rf_3stagev3_20260818_l0p500_a1p400.py"
CONV_BEST="${SCRIPT_DIR}/models/convguided_20260818_l0p500_a1p400/convguided_filter_best_20260818_l0p500_a1p400.pth"
RF_BEST="${SCRIPT_DIR}/models/rf_3stage_20260818_l0p500_a1p400/rf_3stage_best_20260818_l0p500_a1p400.pth"

echo "[1/2] Training ConvGuided"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" conda run --no-capture-output -n "${ENV_NAME}" \
  python -u "${CONV_SCRIPT}"

if [[ ! -f "${CONV_BEST}" ]]; then
  echo "ConvGuided finished without the expected best checkpoint: ${CONV_BEST}" >&2
  exit 1
fi

echo "[2/2] Training RF/refine with the new ConvGuided checkpoint"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" conda run --no-capture-output -n "${ENV_NAME}" \
  python -u "${RF_SCRIPT}"

if [[ ! -f "${RF_BEST}" ]]; then
  echo "RF/refine finished without the expected best checkpoint: ${RF_BEST}" >&2
  exit 1
fi

echo "Training completed."
echo "ConvGuided: ${CONV_BEST}"
echo "RF/refine:  ${RF_BEST}"
