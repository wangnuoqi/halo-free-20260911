#!/usr/bin/env bash
# 下载 5000 个 SVG 源文件；不会生成 PNG/JPEG。再次运行会跳过已有文件。
set -euo pipefail

CLIPART_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CLIPART_PYTHON="${CLIPART_PYTHON:-/home/nuoqi/miniconda3/envs/halofree/bin/python}"

exec "$CLIPART_PYTHON" -u "$CLIPART_PROJECT_DIR/download_openclipart_500.py" \
  --count 5000 \
  --seed 20261006 \
  --output "$CLIPART_PROJECT_DIR/openclipart_5000"
