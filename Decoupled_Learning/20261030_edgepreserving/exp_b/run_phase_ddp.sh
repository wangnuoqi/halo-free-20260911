#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 {base|refine|joint} [extra train_exp_b.py arguments...]" >&2
  exit 2
fi

PHASE="$1"
shift
case "$PHASE" in
  base|refine|joint) ;;
  *) echo "unknown phase: $PHASE" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/runs/exp_b_vivblock_20261102}"
DATA_ROOT="${DATA_ROOT:-/mnt/d/ResearchCode/S2R-HDR/data/20260818}"
OUTPUT_DIR="$RUN_ROOT/$PHASE"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
CROP_SIZE="${CROP_SIZE:-128}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-16}"
EPOCHS="${EPOCHS:-200}"
LR="${LR:-3e-4}"
MIN_LR="${MIN_LR:-3e-6}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PRECISION="${PRECISION:-bf16}"
PIN_MEMORY="${PIN_MEMORY:-0}"
VAL_COUNT="${VAL_COUNT:-200}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
SEED="${SEED:-42}"
MASTER_PORT="${MASTER_PORT:-29641}"
MAX_STEPS="${MAX_STEPS:-0}"
RESUME="${RESUME:-0}"
if [[ "$RESUME" != "0" && "$RESUME" != "1" ]]; then
  echo "RESUME must be 0 or 1" >&2
  exit 2
fi
if [[ "$PIN_MEMORY" != "0" && "$PIN_MEMORY" != "1" ]]; then
  echo "PIN_MEMORY must be 0 or 1" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

BASE_CHECKPOINT="${BASE_CHECKPOINT:-$RUN_ROOT/base/base_best.pth}"
REFINE_CHECKPOINT="${REFINE_CHECKPOINT:-$RUN_ROOT/refine/refine_best.pth}"
if [[ "$PHASE" == "refine" && ! -f "$BASE_CHECKPOINT" ]]; then
  echo "Missing base checkpoint: $BASE_CHECKPOINT" >&2
  exit 2
fi
if [[ "$PHASE" == "joint" && ( ! -f "$BASE_CHECKPOINT" || ! -f "$REFINE_CHECKPOINT" ) ]]; then
  echo "Joint phase requires both checkpoints:" >&2
  echo "  base:   $BASE_CHECKPOINT" >&2
  echo "  refine: $REFINE_CHECKPOINT" >&2
  exit 2
fi
if [[ "$RESUME" == "0" && -f "$OUTPUT_DIR/${PHASE}_latest.pth" ]]; then
  echo "Run output already exists: $OUTPUT_DIR/${PHASE}_latest.pth" >&2
  echo "Choose a new RUN_ROOT or set RESUME=1 for the matching phase." >&2
  exit 2
fi

ARGS=(
  --phase "$PHASE"
  --data-root "$DATA_ROOT"
  --output-dir "$OUTPUT_DIR"
  --crop-size "$CROP_SIZE"
  --batch-size-per-rank "$BATCH_SIZE_PER_GPU"
  --epochs "$EPOCHS"
  --lr "$LR"
  --min-lr "$MIN_LR"
  --weight-decay "$WEIGHT_DECAY"
  --lr-schedule cosine
  --precision "$PRECISION"
  --num-workers "$NUM_WORKERS"
  --val-count "$VAL_COUNT"
  --seed "$SEED"
  --parallelism ddp
)
if [[ "$PIN_MEMORY" == "1" ]]; then
  ARGS+=(--pin-memory)
fi
if [[ "$PHASE" != "base" ]]; then
  ARGS+=(--base-checkpoint "$BASE_CHECKPOINT")
fi
if [[ "$PHASE" == "joint" ]]; then
  ARGS+=(--refine-checkpoint "$REFINE_CHECKPOINT")
fi
if [[ "$MAX_STEPS" != "0" ]]; then
  ARGS+=(--max-steps "$MAX_STEPS")
fi
if [[ "$RESUME" == "1" ]]; then
  ARGS+=(--resume)
fi

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_DIR"
echo "Exp B phase=$PHASE; GPUs=$CUDA_VISIBLE_DEVICES; crop=$CROP_SIZE; batch/GPU=$BATCH_SIZE_PER_GPU; epochs=$EPOCHS; workers/rank=$NUM_WORKERS; output=$OUTPUT_DIR"
"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  --master_port="$MASTER_PORT" \
  exp_b/src/train_exp_b.py "${ARGS[@]}" "$@"
