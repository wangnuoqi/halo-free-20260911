#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_WIN='D:/nuoqi/halo-free/Decoupled_Learning/20260815_edgepreserving'

if declare -F wintrain >/dev/null 2>&1; then
    run_wintrain() { wintrain "$1"; }
elif [[ -x /mnt/d/nuoqi/tools/wintrain_logged.sh ]]; then
    run_wintrain() { /mnt/d/nuoqi/tools/wintrain_logged.sh "$1" "$2"; }
else
    echo "[exp2-exp3] error: wintrain launcher unavailable" >&2
    exit 127
fi

run_one() {
    local name="$1"
    local script="${ROOT_WIN}/experiments/${name}/run_${name}.ps1"
    local payload="\$env:CUDA_VISIBLE_DEVICES='0,1,2,3'; \$env:PYTHONUNBUFFERED='1'; & '${script}'"
    local checkpoint
    local rc
    if [[ "$name" == "exp2_groupnorm" ]]; then
        checkpoint="experiments/exp2_groupnorm/models/exp2_groupnorm_rf_3stage_20260818_l0p500_a1p400/rf_3stage_best_20260818_l0p500_a1p400.pth"
    else
        checkpoint="experiments/exp3_softedge/models/exp3_softedge_rf_3stage_20260818_l0p500_a1p400/rf_3stage_best_20260818_l0p500_a1p400.pth"
    fi
    echo "[$name] start"
    set +e
    run_wintrain "$payload" "edgepreserving_${name}"
    rc=$?
    set -e
    if [[ ! -f "$checkpoint" ]]; then
        echo "[$name] failed: exit=$rc and checkpoint is missing: $checkpoint" >&2
        return "${rc:-1}"
    fi
    if [[ "$rc" -ne 0 ]]; then
        echo "[$name] warning: launcher exit=$rc, but checkpoint exists; continuing"
    fi
    echo "[$name] finished"
}

run_one exp2_groupnorm
run_one exp3_softedge
echo "[exp2-exp3] all requested experiments finished"
