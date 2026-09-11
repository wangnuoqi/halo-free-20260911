#!/usr/bin/env bash
set -Eeuo pipefail

payload='$env:CUDA_VISIBLE_DEVICES="1"; $env:PYTHONUNBUFFERED="1"; conda run --no-capture-output -n yolo_clean python -u D:\nuoqi\halo-free\Decoupled_Learning\20260621_denoiseretrain\exp7\run_exp7.py --data-root "D:\nuoqi\halo-free\data_bsds" --run-root "D:\nuoqi\halo-free\Decoupled_Learning\20260621_denoiseretrain\exp7\formal_v2" --only convguided-joint --full-batch-size 8 --device cuda:0 --num-workers 8; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; conda run --no-capture-output -n yolo_clean python -u D:\nuoqi\halo-free\Decoupled_Learning\20260621_denoiseretrain\exp7\run_exp7.py --data-root "D:\nuoqi\halo-free\data_bsds" --run-root "D:\nuoqi\halo-free\Decoupled_Learning\20260621_denoiseretrain\exp7\formal_v2" --only convguided-frozen --full-batch-size 8 --device cuda:0 --num-workers 8'

if declare -F wintrain >/dev/null 2>&1; then
  wintrain "$payload"
elif [[ -x /mnt/d/nuoqi/tools/wintrain_logged.sh ]]; then
  /mnt/d/nuoqi/tools/wintrain_logged.sh "$payload" denoiser_exp7_convguided_rf_gpu1
else
  echo 'wintrain launcher unavailable' >&2
  exit 127
fi
