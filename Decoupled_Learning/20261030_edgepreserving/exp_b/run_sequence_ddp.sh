#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_PHASE="${START_PHASE:-base}"
RESUME_START="${RESUME:-0}"
case "$START_PHASE" in
  base|refine|joint) ;;
  *) echo "START_PHASE must be base, refine, or joint" >&2; exit 2 ;;
esac
start_reached=0
for PHASE in base refine joint; do
  if [[ "$PHASE" == "$START_PHASE" ]]; then
    start_reached=1
  fi
  if [[ "$start_reached" == "0" ]]; then
    continue
  fi
  phase_resume=0
  if [[ "$PHASE" == "$START_PHASE" ]]; then
    phase_resume="$RESUME_START"
  fi
  echo
  echo "===== Exp B: $PHASE ====="
  RESUME="$phase_resume" bash "$SCRIPT_DIR/run_phase_ddp.sh" "$PHASE" "$@"
done
