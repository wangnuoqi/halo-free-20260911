#!/usr/bin/env bash
set -Eeuo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
bash "$root/run_exp7_dncnn_rf_four_gpu_wintrain.sh"
bash "$root/run_exp7_ffdnet_rf_single_gpu_wintrain.sh"
bash "$root/run_exp7_convguided_rf_single_gpu_wintrain.sh"
