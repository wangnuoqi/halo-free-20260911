#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_WIN='D:/nuoqi/halo-free/Decoupled_Learning/20260815_edgepreserving'
if declare -F wintrain >/dev/null 2>&1; then
    run_wintrain() { wintrain "$1"; }
elif [[ -x /mnt/d/nuoqi/tools/wintrain_logged.sh ]]; then
    run_wintrain() { /mnt/d/nuoqi/tools/wintrain_logged.sh "$1" "$2"; }
else
    echo "[exp5] error: wintrain launcher unavailable" >&2
    exit 127
fi

for name in exp5A exp5B exp5C exp5D exp5E; do
    script="${ROOT_WIN}/experiments/exp5/${name}/run_${name}.ps1"
    case "$name" in
        exp5A) checkpoint="experiments/exp5/exp5A/models/exp5A_groupnorm_weights_005_025_070/rf_3stage_best_20260818_l0p500_a1p400.pth" ;;
        exp5B) checkpoint="experiments/exp5/exp5B/models/exp5B_groupnorm_lr3e3/rf_3stage_best_20260818_l0p500_a1p400.pth" ;;
        exp5C) checkpoint="experiments/exp5/exp5C/models/exp5C_groupnorm_lr1e3/rf_3stage_best_20260818_l0p500_a1p400.pth" ;;
        exp5D) checkpoint="experiments/exp5/exp5D/models/exp5D_groupnorm_identity_init/rf_3stage_best_20260818_l0p500_a1p400.pth" ;;
        exp5E) checkpoint="experiments/exp5/exp5E/models/exp5E_groupnorm_stage3_damped/rf_3stage_best_20260818_l0p500_a1p400.pth" ;;
    esac
    payload="\$env:CUDA_VISIBLE_DEVICES='0,1,2,3'; \$env:PYTHONUNBUFFERED='1'; & '${script}'"
    echo "[$name] start"
    set +e
    run_wintrain "$payload" "edgepreserving_${name}"
    rc=$?
    set -e
    if [[ ! -f "$checkpoint" ]]; then
        echo "[$name] failed: exit=$rc and checkpoint is missing: $checkpoint" >&2
        exit "${rc:-1}"
    fi
    if [[ "$rc" -ne 0 ]]; then
        echo "[$name] warning: launcher exit=$rc, but checkpoint exists; continuing"
    fi
    echo "[$name] finished"
done

echo "[exp5] A-E experiments finished"
