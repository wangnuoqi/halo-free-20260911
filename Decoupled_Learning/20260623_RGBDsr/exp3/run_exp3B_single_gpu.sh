#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RGBDSR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${RGBDSR_ROOT}"
export CUDA_VISIBLE_DEVICES=2

COMMON=(--method dkn
  --origin-checkpoint third_party/DKN/parameter/DKN_8x
  --dkn-data "${RGBDSR_ROOT}/exp1/data/dkn_nyuv2_1000_449"
  --epochs 80 --batch-size 1 --origin-lr 1e-4 --rf-lr 3e-3
  --origin-loss-weight 1.0 --device cuda:0)

conda run --no-capture-output -n halofree python -u exp3/train_exp3_joint.py \
  "${COMMON[@]}" --freeze-origin \
  --output-dir exp3/models/exp3B_dkn_rf_frozen_full1000
conda run --no-capture-output -n halofree python -u exp3/train_exp3_joint.py \
  "${COMMON[@]}" \
  --output-dir exp3/models/exp3B_dkn_rf_joint_full1000
