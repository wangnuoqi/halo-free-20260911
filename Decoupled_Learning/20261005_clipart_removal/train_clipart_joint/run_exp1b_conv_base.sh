#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${CLIPART_PYTHON:-/home/nuoqi/miniconda3/envs/halofree/bin/python}"
exec "$PYTHON" -u "$ROOT/train_clipart_joint/train_exp1b_conv_base.py" --data-root "$ROOT/openclipart_5000/jpeg_pairs_512" --list-dir "$ROOT/openclipart_5000" --quality 30 --epochs 80 --batch-size 1 --output-dir "$ROOT/train_clipart_joint/ckpt/exp1-b_conv_base"
