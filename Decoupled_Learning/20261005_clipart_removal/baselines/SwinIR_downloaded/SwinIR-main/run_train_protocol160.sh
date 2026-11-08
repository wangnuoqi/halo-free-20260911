#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$script_dir/../../.." && pwd)"
output_dir="$script_dir/checkpoints/q30_protocol160"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
mkdir -p "$output_dir"
exec > >(tee -a "$output_dir/train.log") 2>&1

resume_args=()
if [[ -f "$output_dir/last.pth" ]]; then
  resume_args=(--resume "$output_dir/last.pth")
fi

conda run --no-capture-output -n halofree python -u "$script_dir/train_paired.py" \
  --data-root "$project_dir/openclipart_5000/jpeg_pairs_512" \
  --list-dir "$project_dir/openclipart_5000" \
  --quality 30 \
  --output-dir "$output_dir" \
  --epochs 160 \
  --batch-size 8 \
  --micro-batch-size 2 \
  --patch-size 128 \
  --lr 0.0002 \
  --lr-milestones 40 80 120 \
  --val-every 5 \
  --workers 16 \
  "${resume_args[@]}"
