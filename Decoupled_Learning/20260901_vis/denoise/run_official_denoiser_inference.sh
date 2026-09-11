#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/mnt/d/nuoqi/halo-free/data_bsds/BSD68}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/official_sigma25}"
DEVICE="${DEVICE:-cuda:0}"
NOISE_SEED="${NOISE_SEED:-0}"

conda run --no-capture-output -n halofree python -u \
  "$ROOT/evaluation/run_official_denoiser_inference.py" \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --device "$DEVICE" \
  --noise-seed "$NOISE_SEED"
