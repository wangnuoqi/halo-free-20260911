#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_WIN='D:/nuoqi/halo-free/Decoupled_Learning/20260815_edgepreserving'
if declare -F wintrain >/dev/null 2>&1; then
  run_wintrain() { wintrain "$1"; }
elif [[ -x /mnt/d/nuoqi/tools/wintrain_logged.sh ]]; then
  run_wintrain() { /mnt/d/nuoqi/tools/wintrain_logged.sh "$1" "$2"; }
else
  echo '[exp8] wintrain launcher unavailable' >&2
  exit 127
fi

for name in exp8A exp8B; do
  case "$name" in
    exp8A)
      script="${ROOT_WIN}/experiments/exp8A_run.ps1"
      ckpt='experiments/exp8/exp8A_20260813_joint_fidelity/models/exp8A_20260813_joint_fidelity/rf_3stage_best_20260818_l0p500_a1p400.pth'
      ;;
    exp8B)
      script="${ROOT_WIN}/experiments/exp8B_run.ps1"
      ckpt='experiments/exp8/exp8B_mixed_joint_fidelity/models/exp8B_mixed_joint_fidelity/rf_3stage_best_20260818_l0p500_a1p400.pth'
      ;;
  esac
  payload="\$env:CUDA_VISIBLE_DEVICES='0,1,2,3'; \$env:PYTHONUNBUFFERED='1'; & '${script}'"
  echo "[$name] start"
  set +e
  run_wintrain "$payload" "edgepreserving_${name}"
  rc=$?
  set -e
  if [[ ! -f "$ckpt" ]]; then
    echo "[$name] checkpoint missing: $ckpt" >&2
    exit "${rc:-1}"
  fi
  echo "[$name] finished"
done

echo '[exp8] A-B experiments finished'
