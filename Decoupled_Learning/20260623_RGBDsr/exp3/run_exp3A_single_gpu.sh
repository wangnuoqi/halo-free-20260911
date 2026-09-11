#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"
export CUDA_VISIBLE_DEVICES=1

COMMON=(--method conv
  --origin-checkpoint exp1/models/conv_survey_dgf_x8/rgb_guided_depth_x8_best.pth
  --data-root /mnt/d/nuoqi/halo-free/data_NYUv2/NYUv2-resplit/NYUv2
  --test-root /mnt/d/nuoqi/halo-free/data_NYUv2/benchmark_test_449
  --epochs 80 --batch-size 4 --origin-lr 3e-4 --rf-lr 3e-3
  --origin-loss-weight 1.0 --device cuda:0)

conda run --no-capture-output -n halofree python -u exp3/train_exp3_joint.py \
  "${COMMON[@]}" --freeze-origin \
  --output-dir exp3/models/exp3A_convguided_rf_frozen_full1000
conda run --no-capture-output -n halofree python -u exp3/train_exp3_joint.py \
  "${COMMON[@]}" \
  --output-dir exp3/models/exp3A_convguided_rf_joint_full1000
