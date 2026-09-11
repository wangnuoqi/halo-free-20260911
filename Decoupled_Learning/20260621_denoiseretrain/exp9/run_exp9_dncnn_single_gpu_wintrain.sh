#!/usr/bin/env bash
set -Eeuo pipefail
gpu_id="${GPU_ID:-0}"
exp_wsl="$(cd "$(dirname "$0")" && pwd)"
repo_wsl="$(cd "$exp_wsl/.." && pwd)"
repo_win="$(wslpath -w "$repo_wsl")"
data_win="$(wslpath -w "$(cd "$repo_wsl/../.." && pwd)/data_bsds")"
payload="\$env:CUDA_VISIBLE_DEVICES=\"$gpu_id\"; \$env:PYTHONUNBUFFERED=\"1\"; conda run --no-capture-output -n yolo_clean python -u \"$repo_win/exp9/run_exp9.py\" --denoiser dncnn --data-root \"$data_win\" --run-root \"$repo_win/exp9/formal_v1\" --device cuda:0 --num-workers 8"
if declare -F wintrain >/dev/null 2>&1; then wintrain "$payload"; elif [[ -x /mnt/d/nuoqi/tools/wintrain_logged.sh ]]; then /mnt/d/nuoqi/tools/wintrain_logged.sh "$payload" exp9_dncnn; else echo 'wintrain launcher unavailable' >&2; exit 127; fi
