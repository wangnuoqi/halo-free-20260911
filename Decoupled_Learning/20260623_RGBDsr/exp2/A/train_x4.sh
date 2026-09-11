#!/usr/bin/env bash
set -euo pipefail

python train_convguided_base.py --scale 4 --batch-size 8 --epochs 80 "$@"
