#!/usr/bin/env bash
set -euo pipefail

# Runs the four experiments serially. Each refine trainer uses DataParallel
# across all four visible GPUs; the next experiment starts only after success.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/nuoqi/miniconda3/envs/halofree/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/d/nuoqi/halo-free/data_bsds}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GPU_DEVICE="${GPU_DEVICE:-cuda:0}"
CACHE_DEVICE="${CACHE_DEVICE:-cpu}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/third_party/refine_runs/bsds400_bsd68_sigma25}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
mkdir -p "${RUN_ROOT}/logs"
exec > >(tee -a "${RUN_ROOT}/logs/run_$(date +%Y%m%d_%H%M%S).log") 2>&1

echo "[$(date)] GPUs=${CUDA_VISIBLE_DEVICES} cache_device=${CACHE_DEVICE} train_device=${GPU_DEVICE} epochs=${EPOCHS} batch_size=${BATCH_SIZE}"
echo "[$(date)] output=${RUN_ROOT}"

DN_CACHE="${RUN_ROOT}/dncnn_sigma25_cache.pt"
FF_CACHE="${RUN_ROOT}/ffdnet_sigma25_cache.pt"

cache_valid() {
  [[ -s "$1" ]] || return 1
  "${PYTHON_BIN}" - "$1" <<'PY'
import sys, torch
try:
    x=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
    assert x['train'] and x['test'] and x['sigma'] == 25.0
except Exception:
    raise SystemExit(1)
PY
}

if ! cache_valid "${DN_CACHE}"; then
  rm -f "${DN_CACHE}"
  "${PYTHON_BIN}" -u "${ROOT_DIR}/third_party/DnCNN/refine_bridge/generate_cache.py" \
    --train-root "${DATA_ROOT}/BSDS 400" --test-root "${DATA_ROOT}/BSD68" \
    --out "${DN_CACHE}" --sigma 25 --device "${CACHE_DEVICE}"
fi

if ! cache_valid "${FF_CACHE}"; then
  rm -f "${FF_CACHE}"
  "${PYTHON_BIN}" -u "${ROOT_DIR}/third_party/FFDNet/refine_bridge/generate_cache.py" \
    --train-root "${DATA_ROOT}/BSDS 400" --test-root "${DATA_ROOT}/BSD68" \
    --out "${FF_CACHE}" --sigma 25 --device "${CACHE_DEVICE}"
fi

run_one() {
  local denoiser="$1" cache="$2" variant="$3" trainer="$4"
  local out="${RUN_ROOT}/${denoiser}_sigma25_refine_${variant}"
  echo "[$(date)] START ${denoiser} -> ${variant}"
  "${PYTHON_BIN}" -u "${trainer}" --cache "${cache}" --variant "${variant}" \
    --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" --device "${GPU_DEVICE}" --out "${out}"
  echo "[$(date)] DONE ${denoiser} -> ${variant}; model=${out}/${denoiser,,}_sigma25_refine_${variant}_best.pth"
}

run_one DnCNN "${DN_CACHE}" groupnorm "${ROOT_DIR}/third_party/DnCNN/refine_bridge/train_refine.py"
run_one DnCNN "${DN_CACHE}" original  "${ROOT_DIR}/third_party/DnCNN/refine_bridge/train_refine.py"
run_one FFDNet "${FF_CACHE}" groupnorm "${ROOT_DIR}/third_party/FFDNet/refine_bridge/train_refine.py"
run_one FFDNet "${FF_CACHE}" original  "${ROOT_DIR}/third_party/FFDNet/refine_bridge/train_refine.py"

echo "[$(date)] ALL FOUR EXPERIMENTS COMPLETED"
