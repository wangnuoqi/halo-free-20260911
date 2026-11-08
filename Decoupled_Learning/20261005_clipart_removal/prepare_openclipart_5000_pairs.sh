#!/usr/bin/env bash
# 将 openclipart_5000/svg_raw 生成 clean PNG GT 和 JPEG degraded 输入。
set -euo pipefail

CLIPART_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CLIPART_PYTHON="${CLIPART_PYTHON:-/home/nuoqi/miniconda3/envs/halofree/bin/python}"

exec "$CLIPART_PYTHON" -u "$CLIPART_PROJECT_DIR/prepare_clipart_jpeg_pairs.py" \
  --input "$CLIPART_PROJECT_DIR/openclipart_5000/svg_raw" \
  --output "$CLIPART_PROJECT_DIR/openclipart_5000/jpeg_pairs_512" \
  --size 512 \
  --quality 10 20 30 50 \
  --subsampling 2 \
  --background '#ffffff'
