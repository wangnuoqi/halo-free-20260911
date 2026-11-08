#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_dir="$script_dir/checkpoints/q30_protocol160"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export PYTHONUNBUFFERED=1
export USE_LIBUV=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

if [[ ! -f "$output_dir/last.pth" ]]; then
  echo "Missing complete checkpoint: $output_dir/last.pth" >&2
  exit 1
fi

exec > >(tee -a "$output_dir/train_ddp.log") 2>&1
conda run --no-capture-output -n halofree python -u -m torch.distributed.run \
  --standalone --nproc_per_node=3 "$script_dir/train_paired_ddp.py" \
  --resume "$output_dir/last.pth" \
  --output-dir "$output_dir" \
  "$@"
