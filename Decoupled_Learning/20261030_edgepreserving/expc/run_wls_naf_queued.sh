#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
idle_streak=0

echo "Waiting for all four GPUs to remain below 10% utilization."
while (( idle_streak < 3 )); do
    if nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits \
        | awk 'BEGIN {ok=1; n=0} {gsub(/[[:space:]]/, "", $1); n++; if ($1 > 10) ok=0} END {exit !(ok && n==4)}'; then
        idle_streak=$((idle_streak + 1))
        echo "Four-GPU idle probe ${idle_streak}/3 passed at $(date -Is)."
    else
        idle_streak=0
        echo "Four-GPU set is busy at $(date -Is); checking again in 60s."
    fi
    if (( idle_streak < 3 )); then sleep 60; fi
done

cd "$ROOT"
export WINTRAIN_MODE=foreground
exec ./run_expc_wintrain.sh \
    --variant wls_naf \
    --run-root runs/0818_wls_naf_targeted \
    --evaluation-root evaluation/test_20260818 \
    --formal-iterations 60000 \
    --screening-iterations 5000 \
    --batch-per-gpu 16 \
    --num-workers 0 \
    --save-every 100
