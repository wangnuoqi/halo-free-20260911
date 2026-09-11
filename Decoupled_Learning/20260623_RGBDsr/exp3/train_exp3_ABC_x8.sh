#!/usr/bin/env bash
set -euo pipefail

WIN_ROOT='D:\nuoqi\halo-free\Decoupled_Learning\20260623_RGBDsr'
WIN_DATA='D:\nuoqi\halo-free\data_NYUv2\NYUv2-resplit\NYUv2'
WIN_TEST='D:\nuoqi\halo-free\data_NYUv2\benchmark_test_449'
WIN_DKN_DATA='D:\nuoqi\halo-free\Decoupled_Learning\20260623_RGBDsr\exp1\data\dkn_nyuv2_1000_449'

if declare -F wintrain >/dev/null 2>&1; then
  run_wintrain() { wintrain "$1"; }
elif [[ -x /mnt/d/nuoqi/tools/wintrain_logged.sh ]]; then
  run_wintrain() { /mnt/d/nuoqi/tools/wintrain_logged.sh "$1" "$2"; }
else
  echo 'wintrain launcher unavailable' >&2
  exit 127
fi

run_joint() {
  local tag="$1" method="$2" checkpoint="$3" epochs="$4" batch="$5" origin_lr="$6" resume="${7:-}"
  local output_dir="${WIN_ROOT}\\exp3\\models\\${tag}"
  local resume_arg=""
  if [[ -n "${resume}" ]]; then
    resume_arg=" --resume '${resume}'"
  fi
  local payload="\$env:CUDA_VISIBLE_DEVICES='0,1,2,3'; \$env:PYTHONUNBUFFERED='1'; conda run --no-capture-output -n yolo_clean python -u '${WIN_ROOT}\exp3\train_exp3_joint.py' --method '${method}' --origin-checkpoint '${checkpoint}' --data-root '${WIN_DATA}' --test-root '${WIN_TEST}' --dkn-data '${WIN_DKN_DATA}' --output-dir '${output_dir}' --epochs ${epochs} --batch-size ${batch} --origin-lr ${origin_lr} --rf-lr 0.003 --device cuda:0${resume_arg}"
  run_wintrain "$payload" "rgbdsr_${tag}"
}

run_existing_eval() {
  local tag="$1" method="$2" checkpoint="$3" output_dir="$4"
  local payload="\$env:CUDA_VISIBLE_DEVICES='0,1,2,3'; \$env:PYTHONUNBUFFERED='1'; conda run --no-capture-output -n yolo_clean python -u '${WIN_ROOT}\exp3\train_exp3_joint.py' --method '${method}' --origin-checkpoint '${checkpoint}' --data-root '${WIN_DATA}' --test-root '${WIN_TEST}' --dkn-data '${WIN_DKN_DATA}' --output-dir '${output_dir}' --epochs 0 --batch-size 4 --origin-lr 0.0003 --rf-lr 0.003 --device cuda:0"
  run_wintrain "$payload" "rgbdsr_${tag}"
}

# Exp3-A is rerun with the unified 1000-image pool and the Exp7C fidelity plus
# ConvGuided's original L1 origin loss. The previous A checkpoint used 900
# images and the pre-fidelity objective, so it is not reused.
run_joint exp3A_convguided_joint_full1000 conv "${WIN_ROOT}\exp1\models\conv_survey_dgf_x8\rgb_guided_depth_x8_best.pth" 80 4 0.0003
run_joint exp3B_dkn_rf_joint_full80 dkn "${WIN_ROOT}\third_party\DKN\parameter\DKN_8x" 80 1 0.0001
run_joint exp3C_dctnet_rf_joint_full200 dctnet "${WIN_ROOT}\third_party\DCTNet\models\DCTNet_8X.pth" 200 8 0.001
