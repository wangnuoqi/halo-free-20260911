#!/usr/bin/env bash
set -euo pipefail
payload='$env:CUDA_VISIBLE_DEVICES="0,1,2,3"; $env:PYTHONUNBUFFERED="1"; conda run --no-capture-output -n yolo_clean python -u D:\nuoqi\halo-free\Decoupled_Learning\20260621_denoiseretrain\exp7\run_exp7.py --data-root "D:\nuoqi\halo-free\data_bsds" --run-root "D:\nuoqi\halo-free\Decoupled_Learning\20260621_denoiseretrain\exp7\formal_v2" --only joint-rf --device cuda:0 --num-workers 8'
if command -v wintrain >/dev/null 2>&1; then
  wintrain "$payload"
else
  /mnt/d/nuoqi/tools/wintrain_logged.sh "$payload"
fi
