#!/usr/bin/env bash
set -Eeuo pipefail
gpu_id="${GPU_ID:-0}"
payload="\$env:CUDA_VISIBLE_DEVICES=\"$gpu_id\"; \$env:PYTHONUNBUFFERED=\"1\"; conda run --no-capture-output -n yolo_clean python -u D:\nuoqi\halo-free\Decoupled_Learning\20260621_denoiseretrain\exp8\run_exp8.py --data-root \"D:\nuoqi\halo-free\data_bsds\" --run-root \"D:\nuoqi\halo-free\Decoupled_Learning\20260621_denoiseretrain\exp8\formal_v2\" --only base --device cuda:0 --num-workers 8"
if declare -F wintrain >/dev/null 2>&1; then wintrain "$payload"; elif [[ -x /mnt/d/nuoqi/tools/wintrain_logged.sh ]]; then /mnt/d/nuoqi/tools/wintrain_logged.sh "$payload" denoiser_exp8_convguided_base; else echo 'wintrain launcher unavailable' >&2; exit 127; fi
