#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_WIN='D:/nuoqi/halo-free/Decoupled_Learning/20260815_edgepreserving'
if declare -F wintrain >/dev/null 2>&1; then run_wintrain() { wintrain "$1"; }
elif [[ -x /mnt/d/nuoqi/tools/wintrain_logged.sh ]]; then run_wintrain() { /mnt/d/nuoqi/tools/wintrain_logged.sh "$1" "$2"; }
else echo '[exp7] wintrain launcher unavailable' >&2; exit 127; fi
for name in exp7A exp7B exp7C; do
  case "$name" in
    exp7A) script="${ROOT_WIN}/experiments/exp7A_run.ps1"; ckpt='experiments/exp7/exp7A_20260813_joint/models/exp7A_joint_20260813/rf_3stage_best_20260818_l0p500_a1p400.pth' ;;
    exp7B) script="${ROOT_WIN}/experiments/exp7B_run.ps1"; ckpt='experiments/exp7/exp7B_mixed_joint/models/exp7B_mixed_joint/rf_3stage_best_20260818_l0p500_a1p400.pth' ;;
    exp7C) script="${ROOT_WIN}/experiments/exp7C_run.ps1"; ckpt='experiments/exp7/exp7C_20260818_joint_fidelity/models/exp7C_20260818_joint_fidelity/rf_3stage_best_20260818_l0p500_a1p400.pth' ;;
  esac
  payload="\$env:CUDA_VISIBLE_DEVICES='0,1,2,3'; \$env:PYTHONUNBUFFERED='1'; & '${script}'"
  echo "[$name] start"; set +e; run_wintrain "$payload" "edgepreserving_${name}"; rc=$?; set -e
  if [[ ! -f "$ckpt" ]]; then echo "[$name] checkpoint missing: $ckpt" >&2; exit "${rc:-1}"; fi
  echo "[$name] finished"
done
echo '[exp7] A-C experiments finished'
