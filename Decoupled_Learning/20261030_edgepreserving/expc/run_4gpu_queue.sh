#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$ROOT")"
PYTHON_BIN="${PYTHON_BIN:-/home/nuoqi/miniconda3/envs/halofree/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/d/ResearchCode/S2R-HDR/data/20260818}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/0818_four_hypotheses_4gpu}"
mkdir -p "$RUN_ROOT"

echo "Exp C queue waiting for exclusive CUDA access to GPUs ${GPU_LIST}."
echo "Active jobs will not be interrupted. Log root: ${RUN_ROOT}"
idle_streak=0
while (( idle_streak < 3 )); do
    if nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits \
        | awk -F, 'BEGIN {ok=1} {gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $2); if ($1 > 5000 || $2 > 10) ok=0} END {exit !ok}' \
        && CUDA_VISIBLE_DEVICES="$GPU_LIST" "$PYTHON_BIN" -c \
          'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count()==4; [torch.cuda.get_device_name(i) for i in range(4)]' \
          >/dev/null 2>&1; then
        idle_streak=$((idle_streak + 1))
        echo "CUDA idle probe ${idle_streak}/3 passed at $(date -Is)."
    else
        idle_streak=0
        echo "Four-GPU set is busy/unavailable at $(date -Is); retrying in 60s."
    fi
    if (( idle_streak < 3 )); then sleep 60; fi
done

cd "$PROJECT"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
export NCCL_SHM_DISABLE=0
export OMP_NUM_THREADS=1

variants=(naf_small similarity_pyramid lowres_edge unrolled_diffusion)
port=29631
for variant in "${variants[@]}"; do
    out="$RUN_ROOT/$variant"
    echo "Starting 4-GPU smoke test: $variant at $(date -Is)"
    "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=4 --master_port="$port" \
        "$ROOT/src/train_expc.py" --variant "$variant" --data-root "$DATA_ROOT" \
        --output-dir "$RUN_ROOT/smoke/$variant" --crop-size 128 \
        --batch-size-per-gpu 2 --total-iter 12 --val-every 12 --save-every 12 \
        --val-count 8 --num-workers 0 --precision bf16 --lr 3e-4
    port=$((port + 1))
done

for variant in "${variants[@]}"; do
    out="$RUN_ROOT/$variant"
    echo "Starting formal 4-GPU run: $variant at $(date -Is)"
    "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=4 --master_port="$port" \
        "$ROOT/src/train_expc.py" --variant "$variant" --data-root "$DATA_ROOT" \
        --output-dir "$out" --crop-size "${CROP_SIZE:-128}" \
        --batch-size-per-gpu "${BATCH_SIZE_PER_GPU:-16}" \
        --total-iter "${TOTAL_ITER:-60000}" --val-every "${VAL_EVERY:-5000}" \
        --save-every "${SAVE_EVERY:-1000}" --val-count 200 \
        --num-workers "${NUM_WORKERS:-4}" --precision bf16 \
        --lr "${LR:-3e-4}" --min-lr "${MIN_LR:-1e-6}" \
        2>&1 | tee "$RUN_ROOT/${variant}.train.log"
    port=$((port + 1))
    echo "Evaluating untouched test-200 with best validation checkpoint: $variant"
    "$PYTHON_BIN" "$ROOT/src/evaluate_expc.py" \
        --checkpoint "$out/best.pth" --data-root "$DATA_ROOT" \
        --output-dir "$ROOT/evaluation/test_20260818/$variant" --device cuda:0 \
        2>&1 | tee "$RUN_ROOT/${variant}.test.log"
done

echo "Benchmarking existing 0818 NAFNet/VIVNet checkpoints on the same test-200 device."
mkdir -p "$ROOT/evaluation/test_20260818"
"$PYTHON_BIN" "$ROOT/src/benchmark_references.py" \
    2>&1 | tee "$ROOT/evaluation/test_20260818/reference.benchmark.log"

echo "All four Exp C runs and test evaluations completed at $(date -Is)."
