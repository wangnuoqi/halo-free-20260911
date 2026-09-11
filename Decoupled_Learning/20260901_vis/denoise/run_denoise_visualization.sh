#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
EXP8_ROOT="${EXP8_ROOT:-$ROOT/outputs/exp8a_patch_frozen}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-$ROOT/outputs/official_sigma25}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/visualization/exp8a_vs_official}"
SAMPLES="${SAMPLES:-}"
ERROR_PERCENTILE="${ERROR_PERCENTILE:-99.5}"
ERROR_VMAX="${ERROR_VMAX:-}"
DPI="${DPI:-200}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-denoise}"
mkdir -p "$MPLCONFIGDIR"

sample_args=()
if [[ -n "$SAMPLES" ]]; then
  sample_args+=(--samples "$SAMPLES")
fi
error_vmax_args=()
if [[ -n "$ERROR_VMAX" ]]; then
  error_vmax_args+=(--error-vmax "$ERROR_VMAX")
fi

python -u "$ROOT/visualization/render_denoise_comparison.py" \
  --exp8-root "$EXP8_ROOT" \
  --official-root "$OFFICIAL_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --error-percentile "$ERROR_PERCENTILE" \
  --dpi "$DPI" \
  "${sample_args[@]}" \
  "${error_vmax_args[@]}"
