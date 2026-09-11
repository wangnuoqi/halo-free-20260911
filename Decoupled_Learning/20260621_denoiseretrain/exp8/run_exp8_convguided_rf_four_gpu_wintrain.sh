#!/usr/bin/env bash
set -euo pipefail

gpu_id="${GPU_ID:-0}"
exp_wsl="$(cd "$(dirname "$0")" && pwd)"
repo_wsl="$(cd "$exp_wsl/.." && pwd)"
repo_win="$(wslpath -w "$repo_wsl")"
data_win="$(wslpath -w "$(cd "$repo_wsl/../.." && pwd)/data_bsds")"
payload="\$env:CUDA_VISIBLE_DEVICES=\"$gpu_id\"; \$env:PYTHONUNBUFFERED=\"1\"; conda run --no-capture-output -n yolo_clean python -u $repo_win/exp8/run_exp8.py --data-root \"$data_win\" --run-root \"$repo_win/exp8/formal_v2\" --only convguided-rf --device cuda:0 --num-workers 8"

if command -v wintrain >/dev/null 2>&1; then
  wintrain "$payload"
else
  /mnt/d/nuoqi/tools/wintrain_logged.sh "$payload" denoiser_exp8_convguided_rf
fi
