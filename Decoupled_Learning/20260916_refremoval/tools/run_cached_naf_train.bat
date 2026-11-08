@echo off
setlocal

set "CUDA_VISIBLE_DEVICES=0,1,2,3"
set "PYTHONUNBUFFERED=1"
set "LOCAL_RANK="
set "RANK="
set "WORLD_SIZE="
set "PYTHONPATH=D:\ResearchCode\halo-free\Decoupled_Learning\20260916_refremoval;%PYTHONPATH%"

echo CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES%
echo Training backend=single-process DataParallel

D:\anaconda\Scripts\conda.exe run --no-capture-output -n fumo_official python -u ^
  D:\ResearchCode\halo-free\Decoupled_Learning\20260916_refremoval\refine_dataparallel.py ^
  --cache_root D:\ResearchCode\halo-free\Decoupled_Learning\20260916_refremoval\dataset\prepared_refine ^
  --cache_manifest manifest.jsonl ^
  --output_dir D:\ResearchCode\halo-free\Decoupled_Learning\20260916_refremoval\outputs\naf_cached_refine ^
  --resolution 768 ^
  --resize_scale 1.1 ^
  --batch_size 1 ^
  --epochs 10 ^
  --learning_rate 1e-4 ^
  --min_learning_rate 1e-5 ^
  --weight_decay 1e-4 ^
  --checkpointing_steps 1000 ^
  --checkpoints_total_limit 3 ^
  --l1_weight 0.5 ^
  --lpips_weight 0.25 ^
  --grad_weight 0.25

exit /b %ERRORLEVEL%
