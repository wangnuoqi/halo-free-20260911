#!/usr/bin/env bash
set -euo pipefail

# Full-image v24 runs. Each scale gets its own checkpoint/log directory in the
# Python entrypoint, and the next scale starts only after the previous one exits.
PROJECT_WIN='D:\nuoqi\halo-free\Decoupled_Learning\20260623_RGBDsr'
GPU_LIST="${GPU_LIST:-3}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-16}"

# `wintrain` is a function in interactive WSL shells and is not inherited by
# `bash script.sh`. Use the shared launcher directly when the function is absent.
if declare -F wintrain >/dev/null 2>&1; then
    run_wintrain() {
        wintrain "$1"
    }
elif [[ -x /mnt/d/nuoqi/tools/wintrain_logged.sh ]]; then
    run_wintrain() {
        local payload="$1"
        local scale="$2"
        /mnt/d/nuoqi/tools/wintrain_logged.sh "$payload" "v24_x${scale}"
    }
else
    echo "[v24] error: neither wintrain nor /mnt/d/nuoqi/tools/wintrain_logged.sh is available" >&2
    exit 127
fi

run_scale() {
    local scale="$1"
    local payload
    payload='$env:CUDA_VISIBLE_DEVICES="'"${GPU_LIST}"'"; '
    payload+='$env:PYTHONUNBUFFERED="1"; '
    payload+='conda run --no-capture-output -n yolo_clean python -u '
    payload+="${PROJECT_WIN}\train_convguided_base.py"
    payload+=" --scale ${scale} --batch-size ${BATCH_SIZE} --epochs ${EPOCHS}"

    echo "[v24] starting x${scale}: batch=${BATCH_SIZE}, epochs=${EPOCHS}, GPUs=${GPU_LIST}"
    run_wintrain "$payload" "$scale"
    echo "[v24] completed x${scale}"
}

for scale in 4 8 16; do
    run_scale "$scale"
done

echo "[v24] completed x4, x8, and x16"
