#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

evaluate() {
  local name="$1"
  echo "[Exp6] evaluating $name on BSD68" >&2
  CUDA_VISIBLE_DEVICES="$GPU_ID" python ../exp4/evaluate_exp4a_parametric_deep.py --sigma 25 --device cuda:0 \
    --checkpoint "models/$name/val_best.pth" --output-dir "eval_results/$name"
}

echo "[Exp6] start exp6a_late_stage" >&2
CUDA_VISIBLE_DEVICES="$GPU_ID" python train_exp6.py --name exp6a_late_stage --final-mse-weight 0
evaluate exp6a_late_stage
echo "[Exp6] start exp6b_late_stage_mse5" >&2
CUDA_VISIBLE_DEVICES="$GPU_ID" python train_exp6.py --name exp6b_late_stage_mse5 --final-mse-weight 5
evaluate exp6b_late_stage_mse5
echo "[Exp6] start exp6c_late_stage_mse10" >&2
CUDA_VISIBLE_DEVICES="$GPU_ID" python train_exp6.py --name exp6c_late_stage_mse10 --final-mse-weight 10
evaluate exp6c_late_stage_mse10
