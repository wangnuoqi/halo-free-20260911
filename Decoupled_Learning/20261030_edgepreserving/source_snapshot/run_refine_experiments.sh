#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_WIN='D:/nuoqi/halo-free/Decoupled_Learning/20260815_edgepreserving'

# `wintrain` is only a function in interactive shells, so a plain
# `bash run_refine_experiments.sh` needs the executable launcher fallback.
if declare -F wintrain >/dev/null 2>&1; then
    run_wintrain() {
        wintrain "$1"
    }
elif [[ -x /mnt/d/nuoqi/tools/wintrain_logged.sh ]]; then
    run_wintrain() {
        /mnt/d/nuoqi/tools/wintrain_logged.sh "$1" "$2"
    }
else
    echo "[all] error: neither wintrain nor /mnt/d/nuoqi/tools/wintrain_logged.sh is available" >&2
    exit 127
fi

run_experiment() {
    local name="$1"
    local script="${ROOT_WIN}/experiments/${name}/run_${name}.ps1"
    local payload
    payload="\$env:CUDA_VISIBLE_DEVICES='0,1,2,3'; \$env:PYTHONUNBUFFERED='1'; & '${script}'"
    echo "[$name] start"
    run_wintrain "$payload" "edgepreserving_${name}"
    echo "[$name] finished"
}

run_experiment exp1_loss
run_experiment exp2_groupnorm
run_experiment exp3_softedge

echo "[all] refine experiments finished"
