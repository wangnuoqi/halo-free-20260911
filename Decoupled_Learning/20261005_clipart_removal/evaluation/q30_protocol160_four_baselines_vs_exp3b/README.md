# OpenClipart Q30 Protocol160 Evaluation

## Scope

Compare the four completed RGB baselines with the three-stage RGB model on the
same 466-image Q30 test list. All methods use full-image inference and RGB
metrics against the clean PNG targets.

## Checkpoints

| Method | Checkpoint |
| --- | --- |
| ARCNN | `baselines/ARCNN-pytorch/checkpoints/q30_protocol160/best.pth` |
| FBCNN | `baselines/ckpt/fbcnn_rgb_q30_protocol160/fbcnn_best.pth` |
| SwinIR | `baselines/SwinIR_downloaded/SwinIR-main/checkpoints/q30_protocol160/best.pth` |
| VIVNet | `baselines/VIVNet/experiments/OpenClipart-Q30-VIVNet-T-protocol160-3gpu/models/net_g_latest.pth` |
| Three-stage RGB | `train_clipart_joint/ckpt/exp3-b_joint/joint_best.pth` |

## Protocol

- Test list: `openclipart_5000/test_list.txt` (466 pairs).
- Input: `openclipart_5000/jpeg_pairs_512/jpeg_q30`.
- Target: `openclipart_5000/jpeg_pairs_512/clean_png`.
- Inference: full 512x512 image. VIVNet pads bottom/right to a multiple of 32
  and crops output back; SwinIR uses its official window-size padding.
- Metrics: RGB MSE, MAE, PSNR, SSIM, PSNR-B in `[0,1]`.
- Exp3-b half-resolution input uses the same PIL Lanczos resize as its training
  dataset. Its high-resolution input is the original Q30 RGB image.
- ARCNN, FBCNN, SwinIR and VIVNet train on random 126/128-pixel patches;
  Exp3-b trains on its full-image LR/HR pair. This evaluation aligns test data
  and inference resolution, not training geometry or total optimization budget.

## Run

Run in Windows PowerShell after the active GPU training job releases the cards:

```powershell
$env:CUDA_VISIBLE_DEVICES="0,1,2,3"
$env:PYTHONUNBUFFERED="1"
conda run --no-capture-output -n yolo_clean python -u `
  D:\ResearchCode\halo-free\Decoupled_Learning\20261005_clipart_removal\evaluate_all_models.py `
  --project-root D:\ResearchCode\halo-free\Decoupled_Learning\20261005_clipart_removal `
  --output-dir D:\ResearchCode\halo-free\Decoupled_Learning\20261005_clipart_removal\evaluation\q30_protocol160_four_baselines_vs_exp3b `
  --device cuda:0 --batch-size 4 --multi-gpu `
  --methods ARCNN FBCNN SwinIR VIVNet exp3-b
```

The run writes `summary.csv`, `per_image_metrics.csv`, `evaluation_config.json`,
and per-image RGB predictions in method subdirectories.

## Status

- Evaluator smoke test: all five methods loaded their listed checkpoints and
  completed full-image inference on two test images on CPU.
- Full 466-image evaluation: **not yet run**. In this WSL session, CUDA model
  allocation failed with `CUDA-capable device(s) is/are busy or unavailable`.
  The four GPUs were active in a separate training job. CPU timing for SwinIR
  was 47.4 seconds/image in the smoke run, making a CPU full-set run impractical.
- Do not use the two-image smoke metrics as test-set results. The existing
  `evaluation/q30_non_swin_timed` and `evaluation/q30_swinir_4gpu_bs4` reports
  use older baseline checkpoints and are not substitutes for this protocol160
  evaluation.
