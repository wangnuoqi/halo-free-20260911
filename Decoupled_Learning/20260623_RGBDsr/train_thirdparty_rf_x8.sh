#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
  local method="$1" variant="$2" checkpoint="$3"
  local payload="\$env:CUDA_VISIBLE_DEVICES='0,1,2,3'; \$env:PYTHONUNBUFFERED='1'; conda run --no-capture-output -n yolo_clean python -u '${WIN_ROOT}\train_thirdparty_rf.py' --method '${method}' --variant '${variant}' --scale 8 --data-root '${WIN_DATA}' --test-root '${WIN_TEST}' --coarse-checkpoint '${checkpoint}' --output-dir '${WIN_ROOT}\models\thirdparty_${method}_x8_${variant}' --epochs 80 --batch-size 4 --device cuda:0"
  run_wintrain "$payload" "rgbdsr_${method}_x8_${variant}"
}

run_one dkn original "${WIN_ROOT}\third_party\DKN\parameter\DKN_8x"
run_one dkn exp4b    "${WIN_ROOT}\third_party\DKN\parameter\DKN_8x"
run_one dctnet original "${WIN_ROOT}\third_party\DCTNet\models\DCTNet_8X.pth"
run_one dctnet exp4b    "${WIN_ROOT}\third_party\DCTNet\models\DCTNet_8X.pth"
