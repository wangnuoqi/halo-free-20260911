$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "0,1,2,3"
$env:PYTHONUNBUFFERED = "1"
$script = "D:\nuoqi\halo-free\Decoupled_Learning\20260815_edgepreserving\experiments\exp3_softedge\train_rf_exp3_softedge_20260818.py"
Write-Host "[exp3_softedge] Training refine with continuous gradient edge weights"
conda run --no-capture-output -n yolo_clean python -u $script
exit $LASTEXITCODE
