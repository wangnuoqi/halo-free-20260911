#!/usr/bin/env bash
set -Eeuo pipefail

payload='$env:CUDA_VISIBLE_DEVICES="1"; $env:PYTHONUNBUFFERED="1"; $python="D:\nuoqi\halo-free\Decoupled_Learning\20260621_denoiseretrain\exp7\run_exp7.py"; $data="D:\nuoqi\halo-free\data_bsds"; $runroot=(Join-Path (Split-Path $python) "formal_v2"); conda run --no-capture-output -n yolo_clean python -u $python --data-root $data --run-root $runroot --full-batch-size 8 --device cuda:0 --num-workers 8 --only ffdnet-joint; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; conda run --no-capture-output -n yolo_clean python -u $python --data-root $data --run-root $runroot --full-batch-size 8 --device cuda:0 --num-workers 8 --only ffdnet-frozen; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; conda run --no-capture-output -n yolo_clean python -u $python --data-root $data --run-root $runroot --full-batch-size 8 --device cuda:0 --num-workers 8 --only convguided-joint; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; conda run --no-capture-output -n yolo_clean python -u $python --data-root $data --run-root $runroot --full-batch-size 8 --device cuda:0 --num-workers 8 --only convguided-frozen; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }'

if declare -F wintrain >/dev/null 2>&1; then
  wintrain "$payload"
elif [[ -x /mnt/d/nuoqi/tools/wintrain_logged.sh ]]; then
  /mnt/d/nuoqi/tools/wintrain_logged.sh "$payload" denoiser_exp7_ffdnet_convguided_rf_gpu1
else
  echo 'wintrain launcher unavailable' >&2
  exit 127
fi
