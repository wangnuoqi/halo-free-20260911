#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"
CUDA_VISIBLE_DEVICES="$GPU_ID" python train_exp5a_identity_init.py --sigma 25 --device cuda:0
