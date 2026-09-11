#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_WIN='D:/nuoqi/halo-free/Decoupled_Learning/20260815_edgepreserving'
if declare -F wintrain >/dev/null 2>&1; then
  run_wintrain() { wintrain "$1"; }
elif [[ -x /mnt/d/nuoqi/tools/wintrain_logged.sh ]]; then
  run_wintrain() { /mnt/d/nuoqi/tools/wintrain_logged.sh "$1" "$2"; }
else
  echo '[exp6] wintrain launcher unavailable' >&2; exit 127
fi
for name in exp6A exp6B exp6C exp6D exp6E; do
  script="${ROOT_WIN}/experiments/exp6/${name}/run_${name}.ps1"
  case "$name" in
    exp6A) ckpt='experiments/exp6/exp6A/models/exp6A_groupnorm_learnable_edge_gate/rf_3stage_best_20260818_l0p500_a1p400.pth' ;;
    exp6B) ckpt='experiments/exp6/exp6B/models/exp6B_groupnorm_gate_bounded_residual/rf_3stage_best_20260818_l0p500_a1p400.pth' ;;
    exp6C) ckpt='experiments/exp6/exp6C/models/exp6C_groupnorm_shared_refine_cell/rf_3stage_best_20260818_l0p500_a1p400.pth' ;;
    exp6D) ckpt='experiments/exp6/exp6D/models/exp6D_groupnorm_edge_smooth_gate/rf_3stage_best_20260818_l0p500_a1p400.pth' ;;
    exp6E) ckpt='experiments/exp6/exp6E/models/exp6E_joint_convguided_refine/rf_3stage_best_20260818_l0p500_a1p400.pth' ;;
  esac
  payload="\$env:CUDA_VISIBLE_DEVICES='0,1,2,3'; \$env:PYTHONUNBUFFERED='1'; & '${script}'"
  echo "[$name] start"; set +e; run_wintrain "$payload" "edgepreserving_${name}"; rc=$?; set -e
  if [[ ! -f "$ckpt" ]]; then echo "[$name] checkpoint missing: $ckpt" >&2; exit "${rc:-1}"; fi
  echo "[$name] finished"
done
echo '[exp6] A-E experiments finished'
