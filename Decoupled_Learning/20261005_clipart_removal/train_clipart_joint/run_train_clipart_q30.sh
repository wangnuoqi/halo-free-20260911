#!/usr/bin/env bash
set -euo pipefail

CLIPART_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CLIPART_PYTHON="${CLIPART_PYTHON:-/home/nuoqi/miniconda3/envs/halofree/bin/python}"

exec "$CLIPART_PYTHON" -u "$CLIPART_PROJECT_DIR/train_clipart_joint/train_clipart_joint.py" \
  --data-root "$CLIPART_PROJECT_DIR/openclipart_5000/jpeg_pairs_512" \
  --quality 30 \
  --epochs 80 \
  --batch-size 1 \
  --device cuda:0 \
  --output-dir "$CLIPART_PROJECT_DIR/train_clipart_joint/checkpoints/q30"
