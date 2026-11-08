$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "0,1,2,3"
$env:PYTHONUNBUFFERED = "1"

$script = "D:\nuoqi\halo-free\Decoupled_Learning\20260815_edgepreserving\experiments\exp1_loss\train_rf_exp1_loss_20260818.py"
Write-Host "[exp1_loss] Training refine with explicit edge/smooth/gradient loss weights"
Write-Host "[exp1_loss] Test-set model selection is kept unchanged"
conda run --no-capture-output -n yolo_clean python -u $script
exit $LASTEXITCODE
