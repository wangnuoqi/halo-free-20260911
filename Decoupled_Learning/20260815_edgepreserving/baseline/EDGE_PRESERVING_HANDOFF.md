# Edge-Preserving Baseline Handoff

## Scope

This handoff records the single-channel WLS edge-preserving task under
`20260815_edgepreserving`. The input is the normalized one-channel TIFF with
suffix `-log`; the target is the paired WLS-filtered TIFF with suffix
`-WLS-l0p500-a1p400`. The dataset root used by the baseline runs is:

```text
/mnt/d/ResearchCode/S2R-HDR/data/20260818
```

The same deterministic 200-image holdout is selected from the training split
with seed `42`; the independent test split contains 200 images.

## Current Status

### NAFNet baseline

**Verified: trained checkpoint and independent test evaluation exist.**

- Model: `NAFNetLocal`, width `32`, single-channel input/output.
- Checkpoint evaluated: `baseline/runs/nafnet_wls_patch256_nccl_3gpu_bs32/nafnet_wls_best.pth`.
- The best and latest checkpoints currently both point to iteration `60000`.
- The configured schedule was `200000` iterations; therefore this is a tested
  60k checkpoint, not evidence that the full 200k schedule completed.
- Test result: `49.433777 dB` mean PSNR, `1.4757727e-05` mean MSE,
  `0.00218480` mean MAE over 200 test images.

The exact test summary is stored at
`baseline/evaluation/nafnet_wls_test/summary.json`, with per-image values in
`baseline/evaluation/nafnet_wls_test/per_image_metrics.csv`.

### Three-stage internal reference

**Verified: test metrics exist.** The exp7C joint ConvGuided plus 3-stage
model was evaluated on the same 200-image test split:

- Mean PSNR: `47.1140876 dB`.
- Mean MSE: `2.2727207e-05`.
- Predictions were not saved by this metrics-only evaluation.

Source: `baseline/evaluation/exp7c_test/summary.json`.

This is a comparison reference, not a replacement for the NAFNet baseline.
The protocols differ in architecture and training objective, so the numbers
should be reported with their exact checkpoint and evaluation paths.

### VIVNet baseline

**Implemented and training; no final test result yet.** The current run uses:

- Checkpoint: `baseline/runs/vivnet_wls_patch128_nccl_4gpu_bs48/vivnet_wls_latest.pth`.
- Current saved iteration: `1000 / 100000`.
- Four-GPU NCCL DDP, `48` samples per GPU, global effective batch `192`.
- Random paired `128x128` training crops.
- `micro-batch=24` after resuming from the earlier `micro-batch=16` run.
- BF16, AdamW, learning rate `1e-3`, loss `L1 + 0.1 * FFTLoss`.

Do not compare this VIVNet checkpoint as a final baseline until it has a
designated final/best checkpoint and an independent test summary.

## Verified Artifacts

- NAFNet trainer: `baseline/train_nafnet_wls.py`.
- NAFNet launcher: `baseline/run_nafnet_wls_nccl_3gpu_bs32.sh`.
- NAFNet evaluator: `baseline/evaluate_nafnet_wls.py`.
- NAFNet source: `baseline/NAFNet/`.
- Shared WLS pair reader and crop logic: `baseline/wls_data.py`.
- NAFNet training protocol is embedded in both NAFNet checkpoint files under
  the `run_protocol` key.
- NAFNet protocol: single-channel input/output, random `256x256` paired
  patches, three-GPU NCCL DDP, batch `32 x 3` (effective `96`), BF16,
  AdamW `lr=1e-3`, weight decay `1e-3`, betas `(0.9, 0.9)`, cosine schedule,
  `PSNRLoss`, validation every `20000` iterations, save every `5000`.
- NAFNet test geometry: full image, batch-minimum even crop; predictions were
  not saved.
- VIVNet trainer and launcher: `baseline/vivnet_wls/train.py` and
  `baseline/vivnet_wls/run_4gpu_bs48_patch128.sh`.
- VIVNet current protocol is recorded in
  `baseline/runs/vivnet_wls_patch128_nccl_4gpu_bs48/config_used.json` and in
  its checkpoint `run_protocol`.

## Reproduction

Run the NAFNet test metrics only on visible GPU 1 (logical device `cuda:0`):

```bash
cd /mnt/d/ResearchCode/halo-free/Decoupled_Learning/20260815_edgepreserving
CUDA_VISIBLE_DEVICES=1 \
/home/nuoqi/miniconda3/envs/halofree/bin/python -u baseline/evaluate_nafnet_wls.py \
  --data-root /mnt/d/ResearchCode/S2R-HDR/data/20260818 \
  --checkpoint baseline/runs/nafnet_wls_patch256_nccl_3gpu_bs32/nafnet_wls_best.pth \
  --output-dir baseline/evaluation/nafnet_wls_test \
  --device cuda:0 \
  --precision fp32
```

Resume the NAFNet training protocol only when the same output directory and
three-GPU geometry are intended:

```bash
cd /mnt/d/ResearchCode/halo-free/Decoupled_Learning/20260815_edgepreserving
CUDA_VISIBLE_DEVICES=0,1,2 bash baseline/run_nafnet_wls_nccl_3gpu_bs32.sh --resume
```

Continue the current VIVNet run with the current effective batch and patch
size:

```bash
cd /mnt/d/ResearchCode/halo-free/Decoupled_Learning/20260815_edgepreserving
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash baseline/vivnet_wls/run_4gpu_bs48_patch128.sh \
  --micro-batch-size 24 --resume --save-freq 500
```

## Design and Decision Context

Both baselines consume the original normalized grayscale TIFF directly. No
RGB replication or luminance conversion is performed. Training uses random
paired crops; evaluation uses the complete paired test image after the
loader's batch-minimum even trimming (VIVNet additionally pads the bottom and
right edges to a multiple of four for its architecture).

The NAFNet result is currently the strongest verified baseline on this test
protocol (`49.4338 dB`), while exp7C is the verified three-stage reference
(`47.1141 dB`). VIVNet is still a running experiment and must be evaluated
under the same test split before drawing a model comparison.

## Pitfalls and Constraints

- Do not label the NAFNet 60k checkpoint as a completed 200k training run.
- Keep the NAFNet output directory separate from VIVNet output directories.
- A direct resume must preserve data root, crop geometry, optimizer settings,
  precision, and effective batch. Micro-batch may change when the effective
  batch remains unchanged.
- `CUDA_VISIBLE_DEVICES` remaps visible cards to logical ordinals. For a
  single visible card, use `--device cuda:0` in the evaluator.
- The VIVNet 128-patch checkpoint has no final independent test summary yet.

## Open Questions

- Whether VIVNet converges to a better test metric than the verified NAFNet
  60k checkpoint remains unresolved.
- Whether extending NAFNet beyond iteration 60000 improves the independent
  test metric has not been measured.

## Next Actions

1. Let the VIVNet 128-patch run reach a planned checkpoint, then evaluate its
   best or explicitly selected checkpoint with `baseline/vivnet_wls/evaluate.py`.
2. Record the resulting VIVNet `summary.json` path and metric here.
3. If a fair final comparison is needed, document one checkpoint-selection
   rule and apply it to NAFNet, VIVNet, and the three-stage reference.
