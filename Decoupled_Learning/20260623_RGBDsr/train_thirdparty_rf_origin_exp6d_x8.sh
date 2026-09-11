#!/usr/bin/env bash
set -euo pipefail

WIN_ROOT='D:\nuoqi\halo-free\Decoupled_Learning\20260623_RGBDsr'
WIN_DATA='D:\nuoqi\halo-free\data_NYUv2\NYUv2-resplit\NYUv2'
WIN_TEST='D:\nuoqi\halo-free\data_NYUv2\benchmark_test_449'

if declare -F wintrain >/dev/null 2>&1; then
  run_wintrain() { wintrain "$1"; }
elif [[ -x /mnt/d/nuoqi/tools/wintrain_logged.sh ]]; then
  run_wintrain() { /mnt/d/nuoqi/tools/wintrain_logged.sh "$1" "$2"; }
else
  echo 'wintrain launcher unavailable' >&2
  exit 127
fi

run_one() {
  local method="$1" variant="$2" checkpoint="$3" rgb_flag="${4:-}"
  local payload="\$env:CUDA_VISIBLE_DEVICES='0,1,2,3'; \$env:PYTHONUNBUFFERED='1'; conda run --no-capture-output -n yolo_clean python -u '${WIN_ROOT}\train_thirdparty_rf.py' --method '${method}' --variant '${variant}' --scale 8 --data-root '${WIN_DATA}' --test-root '${WIN_TEST}' --coarse-checkpoint '${checkpoint}' --output-dir '${WIN_ROOT}\models\thirdparty_${method}_x8_${variant}_origin' --epochs 80 --batch-size 4 --lr 0.001 --device cuda:0 ${rgb_flag}"
  run_wintrain "$payload" "rgbdsr_${method}_x8_${variant}_origin"
}

# DKN's released checkpoint performs better with RGB order on our aligned test protocol.
run_one dkn original "${WIN_ROOT}\third_party\DKN\parameter\DKN_8x" '--rgb-order'
run_one dkn exp6d    "${WIN_ROOT}\third_party\DKN\parameter\DKN_8x" '--rgb-order'
run_one dctnet original "${WIN_ROOT}\third_party\DCTNet\models\DCTNet_8X.pth"
run_one dctnet exp6d    "${WIN_ROOT}\third_party\DCTNet\models\DCTNet_8X.pth"
