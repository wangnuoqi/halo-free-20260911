#!/usr/bin/env bash
set -euo pipefail

# Launch from WSL where the wintrain shell function is available.  The payload
# itself runs on Windows in yolo_clean and uses all four requested GPUs.
wintrain '$env:CUDA_VISIBLE_DEVICES="0,1,2,3"; $env:PYTHONUNBUFFERED="1"; conda run --no-capture-output -n yolo_clean python -u D:\nuoqi\halo-free\Decoupled_Learning\20260916_refremoval\refine-3stage\exp1\train.py --cache_root D:\nuoqi\halo-free\Decoupled_Learning\20260916_refremoval\dataset\prepared_refine --cache_manifest manifest.jsonl --output_dir D:\nuoqi\halo-free\Decoupled_Learning\20260916_refremoval\refine-3stage\exp1\outputs --resolution 768 --resize_scale 1.1 --batch_size 4 --epochs 10 --learning_rate 1e-4 --min_learning_rate 1e-5 --base_channels 48 --num_blocks 4 --residual_scale 0.20'
