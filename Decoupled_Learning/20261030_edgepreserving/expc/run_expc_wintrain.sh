#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEQUENCE_ARGS="${*:-}"
PS_COMMAND='$env:CUDA_VISIBLE_DEVICES="0,1,2,3"; $env:PYTHONUNBUFFERED="1"; $env:OMP_NUM_THREADS="1"; $env:CUDA_MODULE_LOADING="LAZY"; conda run --no-capture-output -n yolo_clean python -u D:\ResearchCode\halo-free\Decoupled_Learning\20261030_edgepreserving\expc\run_expc_sequence_wintrain.py '"$SEQUENCE_ARGS"

cd "$ROOT"
mkdir -p runs
export WINTRAIN_MODE=foreground
exec /mnt/d/researchcode/tools/wintrain_logged.sh "$PS_COMMAND" "${WINTRAIN_LOG_PREFIX:-expc_0818}"
