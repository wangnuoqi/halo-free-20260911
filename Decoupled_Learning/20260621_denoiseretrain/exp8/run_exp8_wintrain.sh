#!/usr/bin/env bash
set -euo pipefail
wintrain '$env:CUDA_VISIBLE_DEVICES="0,1,2,3"; $env:PYTHONUNBUFFERED="1"; conda run --no-capture-output -n yolo_clean python -u D:\nuoqi\halo-free\Decoupled_Learning\20260621_denoiseretrain\exp8\run_exp8.py --data-root "D:\nuoqi\halo-free\data_bsds" --run-root "D:\nuoqi\halo-free\Decoupled_Learning\20260621_denoiseretrain\exp8\formal_v2" --device cuda:0'
