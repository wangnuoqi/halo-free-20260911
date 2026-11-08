#!/usr/bin/env bash
set -euo pipefail

# Two-GPU learned-kernel A-series runs matched to E1's effective batch.
# Each rank keeps micro-batch 16 and accumulates two batches per optimizer step.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export TORCH_DDP_BACKEND="nccl"
export PYTHONUNBUFFERED="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_MODULE_LOADING="LAZY"
export NCCL_SHM_DISABLE="0"
export OMP_NUM_THREADS="1"

PYTHON="${PYTHON:-/home/nuoqi/miniconda3/envs/halofree/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/d/ResearchCode/S2R-HDR/data/20260818}"
CACHE_PATH="${CACHE_PATH:-expc/runs/0818_memmap_4gpu/train_0818_float32.npy}"
RUN_ROOT="${RUN_ROOT:-expe/runs/a1_a2_e1_matched_2gpu_accum2}"
E1_PROTOCOL="${E1_PROTOCOL:-expe/runs/scratch/viv_gf_e1/seed42_ddpfix/protocol.json}"
MASTER_PORT="${MASTER_PORT:-29566}"

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

run_variant() {
  local variant="$1"
  local output="$RUN_ROOT/$variant"

  if [[ -e "$output/latest.pth" || -e "$output/final.pth" ]] && [[ ! -f "$output/protocol.json" ]]; then
    echo "[error] $output has checkpoints but no protocol.json" >&2
    exit 1
  fi

  "$PYTHON" - "$E1_PROTOCOL" "$output/protocol.json" "$DATA_ROOT" "$variant" <<'PY'
import json
import sys
from pathlib import Path

e1_path, run_path, data_root, variant = sys.argv[1:]
e1 = json.loads(Path(e1_path).read_text(encoding="utf-8"))
expected_e1 = {
    "world_size": 4,
    "batch_size_per_gpu": 16,
    "effective_batch_size": 64,
    "gradient_accumulation": 1,
    "parallelism": "ddp",
    "backend": "nccl",
    "train_crop_size": 128,
    "validation_pairs": 200,
    "precision": "bf16",
    "optimizer": "AdamW",
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "total_iter": 60000,
    "seed": 42,
    "loss_mode": "base",
    "freeze_lr": False,
    "init_lr_checkpoint": None,
    "init_model_checkpoint": None,
    "data_root": str(Path(data_root).resolve()),
}
for key, value in expected_e1.items():
    if e1.get(key) != value:
        raise SystemExit(f"E1 protocol differs from expected reference: {key}: {e1.get(key)!r} != {value!r}")

run_file = Path(run_path)
if run_file.exists():
    run = json.loads(run_file.read_text(encoding="utf-8"))
    expected_run = dict(expected_e1)
    expected_run.update({
        "world_size": 2,
        "batch_size_per_gpu": 16,
        "effective_batch_size": 64,
        "gradient_accumulation": 2,
    })
    for key, value in expected_run.items():
        if run.get(key) != value:
            raise SystemExit(f"Existing {variant} run does not match this two-GPU protocol: {key}: {run.get(key)!r} != {value!r}")
    for key in ("train_ids_sha256", "validation_ids_sha256", "train_pairs", "loss",
                "lr_source", "lr_predictor_depth", "lr_predictor_width",
                "lr_predictor_refinement_blocks", "lr_predictor_topology"):
        if run.get(key) != e1.get(key):
            raise SystemExit(f"Existing {variant} run does not match E1 data/model training reference: {key}")
    if run.get("variant") != variant:
        raise SystemExit(f"Existing run has variant {run.get('variant')!r}, expected {variant!r}")
PY

  if [[ -f "$output/final.pth" ]]; then
    echo "[skip] $variant already has $output/final.pth"
    return
  fi

  local resume_args=()
  if [[ -f "$output/latest.pth" ]]; then
    resume_args+=(--resume)
  fi

  echo "[start] $variant"
  "$PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=2 \
    --master_port "$MASTER_PORT" \
    expc/src/train_expc.py \
    --parallelism ddp \
    --variant "$variant" \
    --data-root "$DATA_ROOT" \
    --cache-path "$CACHE_PATH" \
    --output-dir "$output" \
    --crop-size 128 \
    --batch-size-per-gpu 16 \
    --grad-accumulation 2 \
    --total-iter 60000 \
    --val-count 200 \
    --val-every 5000 \
    --save-every 100 \
    --num-workers 8 \
    --precision bf16 \
    --lr 3e-4 \
    --min-lr 1e-6 \
    --weight-decay 1e-4 \
    --seed 42 \
    --loss-mode base \
    "${resume_args[@]}"
}

run_variant viv_gf_a1_stats
run_variant viv_gf_a2_stats_edge
run_variant viv_gf_a1_direct
run_variant viv_gf_a2_direct_edge
run_variant viv_gf_a1_fixed_boxes

echo "[done] Two-GPU A-series completed (five variants)."
