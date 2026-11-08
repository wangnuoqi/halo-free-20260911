$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "0,1,2,3"
$env:PYTHONUNBUFFERED = "1"
conda run --no-capture-output -n yolo_clean python -u "D:\nuoqi\halo-free\Decoupled_Learning\20260815_edgepreserving\experiments\exp5\exp5B\train_rf_exp5B_20260818.py"
exit $LASTEXITCODE
