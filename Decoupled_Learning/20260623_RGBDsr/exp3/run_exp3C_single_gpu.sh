#!/usr/bin/env bash
set -euo pipefail

if declare -F wintrain >/dev/null 2>&1; then
  run_wintrain() { wintrain "$1"; }
elif [[ -x /mnt/d/nuoqi/tools/wintrain_logged.sh ]]; then
  run_wintrain() { /mnt/d/nuoqi/tools/wintrain_logged.sh "$1" "rgbdsr_exp3C"; }
else
  echo 'wintrain launcher unavailable' >&2
  exit 127
fi

WIN_ROOT='D:\nuoqi\halo-free\Decoupled_Learning\20260623_RGBDsr'
WIN_DATA='D:\nuoqi\halo-free\data_NYUv2\NYUv2-resplit\NYUv2'
WIN_TEST='D:\nuoqi\halo-free\data_NYUv2\benchmark_test_449'

run_variant() {
  local tag="$1" freeze="$2" freeze_arg=""
  [[ "$freeze" == 1 ]] && freeze_arg=' --freeze-origin'
  local output_dir="${WIN_ROOT}\\exp3\\models\\${tag}"
  local payload="\$env:CUDA_VISIBLE_DEVICES='3'; \$env:PYTHONUNBUFFERED='1'; conda run --no-capture-output -n yolo_clean python -u '${WIN_ROOT}\exp3\train_exp3_joint.py' --method dctnet --origin-checkpoint '${WIN_ROOT}\third_party\DCTNet\models\DCTNet_8X.pth' --dct-train '${WIN_DATA}' --dct-test '${WIN_TEST}' --output-dir '${output_dir}' --epochs 200 --batch-size 8 --origin-lr 1e-3 --rf-lr 3e-3 --origin-loss-weight 1.0 --device cuda:0${freeze_arg}"
  run_wintrain "$payload"
}

run_variant exp3C_dctnet_rf_frozen_full1000 1
run_variant exp3C_dctnet_rf_joint_full1000 0
