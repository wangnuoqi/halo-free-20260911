#!/usr/bin/env bash
set -euo pipefail

TRAIN_ROOT="/mnt/d/nuoqi/halo-free/data_NYUv2/NYUv2-resplit/NYUv2"
TEST_ROOT="/mnt/d/nuoqi/halo-free/data_NYUv2/benchmark_test_449"
BATCH_SIZE=16
EPOCHS=80

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

train_and_test() {
  local scale="$1"
  python train_convguided_base.py --scale "$scale" --train-root "$TRAIN_ROOT" --test-root "$TEST_ROOT" --batch-size "$BATCH_SIZE" --epochs "$EPOCHS"
  python test_convguided_base.py --checkpoint "models/conv_survey_dgf_x${scale}/rgb_guided_depth_x${scale}_best.pth" --data-root "$TEST_ROOT" --scale "$scale" --output-dir "test_results/conv_survey_dgf_x${scale}"
}

train_and_test 8
train_and_test 16
