#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VARIANT="${1:?variant is required}"
RUN_ROOT="${RUN_ROOT:-runs/0818_memmap_4gpu_parallel_v2}"
LOCK_FILE="${ROOT}/runs/expc_4gpu_no_pin.queue.lock"
mkdir -p "$(dirname "$LOCK_FILE")"
exec 9>"$LOCK_FILE"
flock 9
idle_streak=0

echo "[$VARIANT] waiting for all four GPUs to be available; pin_memory=False"
while (( idle_streak < 3 )); do
    if nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits \
        | awk -F, 'BEGIN {ok=1; n=0} {gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $2); n++; if ($1 > 5000 || $2 > 10) ok=0} END {exit !(ok && n==4)}'; then
        idle_streak=$((idle_streak + 1))
        echo "[$VARIANT] idle probe ${idle_streak}/3 passed at $(date -Is)"
    else
        idle_streak=0
        echo "[$VARIANT] GPUs busy at $(date -Is); retrying in 60s"
    fi
    if (( idle_streak < 3 )); then sleep 60; fi
done

cd "$ROOT"
export WINTRAIN_MODE=foreground
exec ./run_expc_wintrain.sh \
    --variant "$VARIANT" \
    --run-root "$RUN_ROOT" \
    --evaluation-root evaluation/test_20260818 \
    --formal-iterations 60000 \
    --screening-iterations 5000 \
    --batch-per-gpu 16 \
    --num-workers 4 \
    --save-every 100
