#!/usr/bin/env bash
set -euo pipefail

WIN_ROOT='D:\\nuoqi\\halo-free\\Decoupled_Learning\\20260623_RGBDsr'
WIN_DKN_DATA='D:\\nuoqi\\halo-free\\Decoupled_Learning\\20260623_RGBDsr\\exp1\\data\\dkn_nyuv2_1000_449'
WIN_DKN_CKPT="${WIN_ROOT}\\third_party\\DKN\\parameter\\DKN_8x"
WIN_DCT_CKPT="${WIN_ROOT}\\third_party\\DCTNet\\models\\DCTNet_8X.pth"
# Set these to DCTNet's official processed H5/test directories before running.
WIN_DCT_TRAIN='D:\\path\\to\\NYU_Train_imgsize_256_scale_8.h5'
WIN_DCT_TEST='D:\\path\\to\\NYU_Test_AfterProcessing_8X'

if declare -F wintrain >/dev/null 2>&1; then
  run_wintrain() { wintrain "$1"; }
elif [[ -x /mnt/d/nuoqi/tools/wintrain_logged.sh ]]; then
  run_wintrain() { /mnt/d/nuoqi/tools/wintrain_logged.sh "$1" "$2"; }
else
  echo 'wintrain launcher unavailable' >&2
  exit 127
fi

run_job() {
  local tag="$1" method="$2" checkpoint="$3" epochs="$4" batch="$5" origin_lr="$6" freeze="$7"
  local output="${WIN_ROOT}\\exp3\\models\\${tag}"
  local freeze_arg=''
  [[ "$freeze" == '1' ]] && freeze_arg=' --freeze-origin'
  local dct_args=''
  if [[ "$method" == 'dctnet' ]]; then
    dct_args=" --dct-train '${WIN_DCT_TRAIN}' --dct-test '${WIN_DCT_TEST}'"
  fi
  local payload="\$env:CUDA_VISIBLE_DEVICES='0,1,2,3'; \$env:PYTHONUNBUFFERED='1'; conda run --no-capture-output -n yolo_clean python -u '${WIN_ROOT}\\exp3\\train_exp3_joint.py' --method '${method}' --origin-checkpoint '${checkpoint}' --dkn-data '${WIN_DKN_DATA}' --output-dir '${output}' --epochs ${epochs} --batch-size ${batch} --origin-lr ${origin_lr} --rf-lr 0.003 --origin-loss-weight 1.0 --device cuda:0${freeze_arg}${dct_args}"
  run_wintrain "$payload" "rgbdsr_${tag}"
}

# Frozen origin: RF only, unified RF learning rate 3e-3.
run_job exp3B_dkn_rf_frozen_full1000 dkn "${WIN_DKN_CKPT}" 80 1 0.0001 1
# Joint origin+RF: DKN official origin lr 1e-4; RF lr remains independently set.
run_job exp3B_dkn_rf_joint_full1000 dkn "${WIN_DKN_CKPT}" 80 1 0.0001 0
# Frozen origin: RF only, unified RF learning rate 3e-3.
run_job exp3C_dctnet_rf_frozen_full1000 dctnet "${WIN_DCT_CKPT}" 200 8 0.001 1
# Joint origin+RF: DCTNet official origin lr 1e-3; RF lr remains independently set.
run_job exp3C_dctnet_rf_joint_full1000 dctnet "${WIN_DCT_CKPT}" 200 8 0.001 0
