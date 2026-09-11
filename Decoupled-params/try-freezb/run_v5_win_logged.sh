#!/usr/bin/env bash
set -euo pipefail

ROOT_WSL="/mnt/d/nuoqi/halo-free/Decoupled-params/try-freezb"
ROOT_WIN="$(wslpath -w "$ROOT_WSL")"
LOG_DIR_WSL="${ROOT_WSL}/logs"

mkdir -p "$LOG_DIR_WSL"

timestamp="$(date +%Y%m%d-%H%M%S)"
log_file_wsl="${LOG_DIR_WSL}/train_v5_${timestamp}.log"
pid_file_wsl="${LOG_DIR_WSL}/train_v5_${timestamp}.pid"
log_file_win="$(wslpath -w "$log_file_wsl")"
pid_file_win="$(wslpath -w "$pid_file_wsl")"

train_batch_per_gpu="${TRAIN_BATCH_PER_GPU:-24}"
test_batch="${TEST_BATCH:-4}"
train_workers="${TRAIN_WORKERS:-4}"
test_workers="${TEST_WORKERS:-2}"
epochs="${EPOCHS:-80}"
visible_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

touch "$log_file_wsl"

inner_ps_script=$(cat <<EOF
Set-Location '$ROOT_WIN'
\$env:CUDA_VISIBLE_DEVICES='$visible_devices'
\$env:NCCL_P2P_DISABLE='1'
\$env:NCCL_IB_DISABLE='1'
\$env:NCCL_ASYNC_ERROR_HANDLING='1'
\$env:PYTORCH_CUDA_ALLOC_CONF='backend:cudaMallocAsync'
conda run --no-capture-output -n yolo_clean python -u .\\train_convguided_base_paraminput_freezeb_v5.py --train-batch-per-gpu $train_batch_per_gpu --test-batch $test_batch --train-workers $train_workers --test-workers $test_workers --epochs $epochs 2>&1 | Tee-Object -FilePath '$log_file_win' -Append
EOF
)

launcher_ps_script=$(cat <<EOF
\$logPath = '$log_file_win'
\$pidPath = '$pid_file_win'
Set-Content -Path \$logPath -Value ''
\$encodedInner = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes(@"
$inner_ps_script
"@))
\$proc = Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile', '-EncodedCommand', \$encodedInner) -WindowStyle Hidden -PassThru
Set-Content -Path \$pidPath -Value \$proc.Id
Write-Output ('Started Windows training. PID=' + \$proc.Id)
Write-Output ('Log file: ' + \$logPath)
Write-Output ('PID file: ' + \$pidPath)
EOF
)

encode_ps() {
  printf '%s' "$1" | iconv -f UTF-8 -t UTF-16LE | base64 -w 0
}

launcher_encoded="$(encode_ps "$launcher_ps_script")"

/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -EncodedCommand "$launcher_encoded"

echo "Tailing ${log_file_wsl}"
echo "Press Ctrl-C to stop tailing without stopping training."
echo "Use the PID in ${pid_file_wsl} if you want to stop the Windows process later."

trap 'echo; echo "Stopped tailing. Training is still running unless you stop the Windows PID."; echo "Log: '"$log_file_wsl"'"; echo "PID: '"$pid_file_wsl"'"; exit 130' INT TERM

tail -n +1 -f "$log_file_wsl"
