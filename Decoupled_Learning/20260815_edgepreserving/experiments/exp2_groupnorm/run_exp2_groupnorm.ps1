$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "0,1,2,3"
$env:PYTHONUNBUFFERED = "1"
$script = "D:\nuoqi\halo-free\Decoupled_Learning\20260815_edgepreserving\experiments\exp2_groupnorm\train_rf_exp2_groupnorm_20260818.py"
Write-Host "[exp2_groupnorm] Training refine with GroupNorm"
conda run --no-capture-output -n yolo_clean python -u $script
exit $LASTEXITCODE
