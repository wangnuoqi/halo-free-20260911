#!/usr/bin/env bash
set -euo pipefail

# Run from any directory. Each pair is independent and writes to its own
# directory, so the script can be resumed by commenting out completed calls.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RGBDSR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${RGBDSR_ROOT}"

GPU="${GPU:-0}"
ENV_NAME="${ENV_NAME:-halofree}"
export CUDA_VISIBLE_DEVICES="${GPU}"

DATA_ROOT="${DATA_ROOT:-/mnt/d/nuoqi/halo-free/data_NYUv2/NYUv2-resplit/NYUv2}"
TEST_ROOT="${TEST_ROOT:-/mnt/d/nuoqi/halo-free/data_NYUv2/benchmark_test_449}"
DKN_DATA="${DKN_DATA:-${RGBDSR_ROOT}/exp1/data/dkn_nyuv2_1000_449}"
DCT_TRAIN="${DCT_TRAIN:-${RGBDSR_ROOT}/exp3/data/NYU_Train_imgsize_256_scale_8.h5}"
DCT_TEST="${DCT_TEST:-${RGBDSR_ROOT}/exp3/data/NYU_Test_AfterProcessing_8X}"

run() {
  conda run --no-capture-output -n "${ENV_NAME}" python -u exp3/train_exp3_joint.py "$@"
}

COMMON=(--data-root "${DATA_ROOT}" --test-root "${TEST_ROOT}"
  --epochs 80 --origin-lr 3e-4 --rf-lr 3e-3 --origin-loss-weight 1.0
  --device cuda:0)

# Exp3-A: ConvGuided origin + Exp6D RF, frozen and jointly trainable.
run --method conv --origin-checkpoint exp1/models/conv_survey_dgf_x8/rgb_guided_depth_x8_best.pth \
  "${COMMON[@]}" --batch-size 4 --freeze-origin \
  --output-dir exp3/models/exp3A_convguided_rf_frozen_full1000
run --method conv --origin-checkpoint exp1/models/conv_survey_dgf_x8/rgb_guided_depth_x8_best.pth \
  "${COMMON[@]}" --batch-size 4 \
  --output-dir exp3/models/exp3A_convguided_rf_joint_full1000

# Exp3-B: official DKN origin + Exp6D RF, frozen and jointly trainable.
DKN_COMMON=(--method dkn --origin-checkpoint third_party/DKN/parameter/DKN_8x
  --dkn-data "${DKN_DATA}" --epochs 80 --batch-size 1 --origin-lr 1e-4
  --rf-lr 3e-3 --origin-loss-weight 1.0 --device cuda:0)
run "${DKN_COMMON[@]}" --freeze-origin \
  --output-dir exp3/models/exp3B_dkn_rf_frozen_full1000
run "${DKN_COMMON[@]}" \
  --output-dir exp3/models/exp3B_dkn_rf_joint_full1000

# Exp3-C: official DCTNet processed data + Exp6D RF, frozen and jointly trainable.
if [[ ! -f "${DCT_TRAIN}" || ! -d "${DCT_TEST}" ]]; then
  echo "Missing DCTNet processed data: ${DCT_TRAIN} or ${DCT_TEST}" >&2
  echo "Set DCT_TRAIN/DCT_TEST or run tools/prepare_dctnet_processed.py first." >&2
  exit 2
fi
DCT_COMMON=(--method dctnet --origin-checkpoint third_party/DCTNet/models/DCTNet_8X.pth
  --dct-train "${DCT_TRAIN}" --dct-test "${DCT_TEST}" --epochs 200 --batch-size 8
  --origin-lr 1e-3 --rf-lr 3e-3 --origin-loss-weight 1.0 --device cuda:0)
run "${DCT_COMMON[@]}" --freeze-origin \
  --output-dir exp3/models/exp3C_dctnet_rf_frozen_full1000
run "${DCT_COMMON[@]}" \
  --output-dir exp3/models/exp3C_dctnet_rf_joint_full1000

echo "Exp3 A/B/C frozen+joint runs completed. Metrics are in each output-dir/test_metrics.json."
