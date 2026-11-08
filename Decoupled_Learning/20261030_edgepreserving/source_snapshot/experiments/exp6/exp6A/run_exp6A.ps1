$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "0,1,2,3"
$env:PYTHONUNBUFFERED = "1"
conda run --no-capture-output -n yolo_clean python -u "D:\nuoqi\halo-free\Decoupled_Learning\20260815_edgepreserving\experiments\exp6\exp6A\train_rf_exp6A_20260818.py"
exit $LASTEXITCODE
