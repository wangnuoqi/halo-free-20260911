#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_WIN='D:/nuoqi/halo-free/Decoupled_Learning/20260815_edgepreserving'
if declare -F wintrain >/dev/null 2>&1; then
    run_wintrain() { wintrain "$1"; }
elif [[ -x /mnt/d/nuoqi/tools/wintrain_logged.sh ]]; then
    run_wintrain() { /mnt/d/nuoqi/tools/wintrain_logged.sh "$1" "$2"; }
else
    echo "[exp4] error: wintrain launcher unavailable" >&2
    exit 127
fi

for name in exp4A exp4B exp4C exp4D exp4E; do
    script="${ROOT_WIN}/experiments/exp4/${name}/run_${name}.ps1"
    checkpoint="experiments/exp4/${name}/models/"
    case "$name" in
        exp4A) checkpoint+="exp4A_groupnorm_original_loss/rf_3stage_best_20260818_l0p500_a1p400.pth" ;;
        exp4B) checkpoint+="exp4B_groupnorm_deep_supervision/rf_3stage_best_20260818_l0p500_a1p400.pth" ;;
        exp4C) checkpoint+="exp4C_groupnorm_final_loss/rf_3stage_best_20260818_l0p500_a1p400.pth" ;;
        exp4D) checkpoint+="exp4D_groupnorm_final_gradient/rf_3stage_best_20260818_l0p500_a1p400.pth" ;;
        exp4E) checkpoint+="exp4E_groupnorm_final_gradient_smooth_penalty/rf_3stage_best_20260818_l0p500_a1p400.pth" ;;
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

echo "[exp4] A-E experiments finished"
