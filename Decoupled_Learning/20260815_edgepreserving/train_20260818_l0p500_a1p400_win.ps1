$ErrorActionPreference = "Stop"

$projectRoot = "D:\nuoqi\halo-free\Decoupled_Learning\20260815_edgepreserving"
$envName = if ($env:ENV_NAME) { $env:ENV_NAME } else { "yolo_clean" }
$convScript = Join-Path $projectRoot "train_convguided_20260818_l0p500_a1p400.py"
$rfScript = Join-Path $projectRoot "train_rf_3stagev3_20260818_l0p500_a1p400.py"
$convBest = Join-Path $projectRoot "models\convguided_20260818_l0p500_a1p400\convguided_filter_best_20260818_l0p500_a1p400.pth"
$rfBest = Join-Path $projectRoot "models\rf_3stage_20260818_l0p500_a1p400\rf_3stage_best_20260818_l0p500_a1p400.pth"

Write-Host "[1/2] Training ConvGuided"
conda run --no-capture-output -n $envName python -u $convScript
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if (-not (Test-Path $convBest)) { throw "ConvGuided finished without the expected best checkpoint: $convBest" }

Write-Host "[2/2] Training RF/refine with the new ConvGuided checkpoint"
conda run --no-capture-output -n $envName python -u $rfScript
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if (-not (Test-Path $rfBest)) { throw "RF/refine finished without the expected best checkpoint: $rfBest" }

Write-Host "Training completed."
Write-Host "ConvGuided: $convBest"
Write-Host "RF/refine:  $rfBest"
