#!/usr/bin/env bash
set -Eeuo pipefail

# Run the three independent Exp7 full-image base jobs concurrently.
# Four slots are exposed; the fourth remains available because Exp7 has
# three base models.
wintrain '$env:PYTHONUNBUFFERED="1"; conda run --no-capture-output -n yolo_clean python -u D:\nuoqi\halo-free\Decoupled_Learning\20260621_denoiseretrain\run_formal_parallel.py --data-root "D:\nuoqi\halo-free\data_bsds" --exp7-root "D:\nuoqi\halo-free\Decoupled_Learning\20260621_denoiseretrain\exp7\formal_v2" --experiments exp7 --gpus 0,1,2,3 --num-workers 8'
