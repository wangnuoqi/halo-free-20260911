# Texture Removal Operator Dataset

The recommended entry point is the MATLAB generator, because the original CELiNet/DecoupleLearning operators and experiment chain are MATLAB. The Python script is retained as a small prototype for quick checks.

Both generators implement the same ordinary-image idea:

```text
ordinary image -> float image in [0, 1] -> classical operator -> supervised label
```

There is deliberately no HDR `log2` dynamic-range conversion or per-image log normalization. The WLS edge weights still use the standard operator definition `log(I + eps)` internally, as in the original MATLAB implementation; this is part of WLS itself and does not turn an ordinary image into HDR data. The default `--mode y` converts RGB images to the Y channel, which matches the historical single-channel WLS training setup.

## One-click MATLAB run

The VOC2012 trainval images and the original MATLAB RTV implementation are already located relative to the project. To generate the fixed-strength RTV dataset, open `run_generate_rtv_voc2012.m` in MATLAB and click **Run**. It resolves all paths from its own file location, checks that the 11,540 expected VOC2012 images and `tsmooth.m` exist, then writes the dataset to:

```text
<halo-free>/datasets/VOC2012/RTV_fixed_lambda_0p01/
```

The run uses one Y-channel target per source image with `lambda=0.01`, `sigma=3`, `sharpness=0.02`, and `maxIter=4`; it creates a deterministic 80/10/10 train/validation/test split using seed `20261008`. It starts a local MATLAB pool with **24 workers** and uses `parfor` to process images in parallel. It requires MATLAB with Image Processing Toolbox and Parallel Computing Toolbox. The full run processes 11,540 images, so RTV generation may take a while. If the output directory already contains files, the script stops instead of overwriting them.

For a color-model experiment, run `run_generate_rtv_voc2012_rgb.m` in MATLAB. It applies the same RTV settings and split seed to the RGB source images, filtering each color channel with the original `tsmooth.m`, and writes float32 RGB TIFF pairs to `RTV_RGB_fixed_lambda_0p01/`. This is a separate dataset; the existing Y-channel dataset is not modified. The RGB dataset requires RGB-capable training models and must not be loaded by the current single-channel `RTVDataset`.

The generic generator can also be called directly for another image directory:

```matlab
generate_operator_dataset_matlab( ...
  'inputDir', '/path/to/ordinary_images', ...
  'outputDir', '/path/to/texture_rtv_dataset', ...
  'operator', 'rtv', ...
  'mode', 'y', ...
  'variantsPerImage', 1, ...
  'valRatio', 0.1, ...
  'testRatio', 0.1, ...
  'seed', 20261008, ...
  'rtvLambda', 0.01, ...
  'rtvSigma', 3.0, ...
  'rtvSharpness', 0.02, ...
  'rtvMaxIter', 4, ...
  'operatorRoot', '/path/to/DecoupleLearning-master');
```

The MATLAB generator defaults to one deterministic variant per image. Keep all RTV parameters fixed in this single-task training set.

The supported MATLAB operators are `wls`, `l0`, `rtv`, `rolling_guidance`, `gaussian`, and `bilateral`. The first four use the original CELiNet MATLAB implementations when `operatorRoot` is supplied. `mode='y'` is the recommended setting for the historical single-channel network; use `mode='rgb'` only when the training network and loss are configured for three channels.

The MATLAB generator writes single-channel 32-bit floating-point TIFF input/label pairs by default for `mode='y'`.

## Three-stage network training

After the MATLAB generation finishes, the network training is split into the same three stages used by the edge-preserving experiments:

1. `train_conv_base_rtv.py` trains the ConvGuided base network.
2. `train_refine_rtv.py` loads the best base checkpoint, freezes it, and trains the three-stage refine network.
3. `train_joint_rtv.py` loads both best checkpoints and jointly fine-tunes them.

The training scripts currently default to the RGB dataset at `RTV_RGB_fixed_lambda_0p01`, and the models are configured for three channels. The separate `RTV_fixed_lambda_0p01` dataset is Y-channel data and cannot be used with these defaults without changing the model constructors. Each generated pair is full resolution. Validation uses one image at a time and evaluates the full image (trimming an odd bottom row or right column to keep dimensions even). Training batches may be truncated to the smallest image dimensions in that batch by `collate_pairs`.

From this directory, run:

```bash
python train_conv_base_rtv.py
python train_refine_rtv.py
python train_joint_rtv.py
```

The scripts default to:

```text
data:   /mnt/d/ResearchCode/halo-free/datasets/VOC2012/RTV_RGB_fixed_lambda_0p01
models: 20261008_textureremove/models/{conv_base_rtv,refine_rtv,joint_rtv}
```

The base checkpoint required by stage 2 is `models/conv_base_rtv/conv_base_rtv_best.pth`; the refine checkpoint required by stage 3 is `models/refine_rtv/refine_rtv_best.pth`. Use `--data-root`, `--epochs`, `--batch-size`, or checkpoint arguments to override defaults. A CUDA GPU is recommended; the scripts fall back to CPU for a smoke test.

## Official DecoupleLearning baseline

The official single-operator model baseline uses the same RGB RTV pairs and split, and now matches the three-stage data loader by default: batch size 8, shuffle enabled, no dropped final batch, BGR channel order, and batch-level top-left trimming to the smallest even height/width. Validation uses batch size 1 on each full image. The 9,232 training images divide evenly into 1,154 batches of 8. New weights go to `baseline/runs/rtv_full_image/`. The existing `baseline/runs/rtv/` checkpoint is the historical fixed-crop, RGB-order run and must not be used as the aligned result. See [baseline/README.md](baseline/README.md) for the command and crop-mode options.

## Unified full-image test evaluation

`eval_rtv_models.py` evaluates the completed Stage 1, Stage 2, Stage 3 joint,
aligned official RTV, and NAFNet patch-256 best checkpoints on exactly the
same RGB RTV test pairs. The default run saves each model's display PNGs,
per-image metrics, `summary.csv`, `protocol.json`, and labeled side-by-side
previews for the first 16 sorted test images. The incomplete NAFNet full-image
checkpoint and historical crop-trained official checkpoint are excluded.

From this directory, first try one image, then run the full test split:

```bash
/home/nuoqi/miniconda3/envs/halofree/bin/python eval_rtv_models.py \
  --device cuda:0 --max-images 1 --output-dir eval_results/smoke
/home/nuoqi/miniconda3/envs/halofree/bin/python eval_rtv_models.py \
  --device cuda:0 --output-dir eval_results/all_test
```

Use `--device cpu` if CUDA is unavailable. Set `--models stage1 stage2 stage3
official nafnet_patch` to change or reorder models, `--repeats 3` to time each
image three times and report its median, and `--no-save-images` if display PNGs
are not needed. For a quick subset, pass `--max-images N` and a separate output
directory. Each run requires a new or empty output directory so results from
different model selections cannot be mixed.

All models receive the same BGR `[0,1]` test tensors, top-left cropped only
to remove an odd bottom row/right column. Predictions are clamped to `[0,1]`;
MSE and MAE are pixel means per image, then averaged equally across images.
The reported inference time starts after host-to-device transfer, includes
each model's required on-device preprocessing, forward pass, and output scaling,
and ends before metric computation or PNG saving. CUDA is synchronized on both
timing boundaries. Model loading and disk I/O are excluded. All models run in
FP32 for this evaluation. PNGs are 8-bit display copies; metrics use float
tensors. The test set is an internal split of VOC2012 trainval, not the
official VOC2012 test set.

Training geometry is **not identical across every checkpoint**: our three
stages and the aligned official model use batch-minimum top-left cropping and
even-size trimming. The completed NAFNet run trains on random paired
`256x256` patches, with an effective batch size of 24 across three GPUs;
it only shares the full-image validation/test geometry. The incomplete
NAFNet full-image run uses batch-minimum trimming within each GPU's local
batch of eight, not the same global batch grouping as our stages.

Python prototype
----------------

```bash
python generate_operator_dataset.py \
  --input-dir /path/to/images \
  --output-dir /path/to/texture_wls_dataset \
  --operator wls \
  --mode y \
  --variants-per-image 7 \
  --output-format tiff
```

The output is split by source image before generating variants:

```text
output/
  train/input/*.tiff
  train/label/*.tiff
  val/input/*.tiff
  val/label/*.tiff
  test/input/*.tiff
  test/label/*.tiff
  metadata.jsonl
```

Each metadata record contains the source image, split, operator, parameters, mode, and paired paths. This prevents variants of the same source image from leaking across splits and makes the generated labels auditable.

Supported operators in the Python prototype:

- `wls`: weighted-least-squares smoothing solved with SciPy. In `--mode rgb`, it is applied independently to the three channels; the default `--mode y` remains the recommended path for a single-channel network.
- `gaussian`: Gaussian smoothing.
- `bilateral`: OpenCV bilateral filtering.
- `rolling_guidance`: repeated bilateral-guidance approximation.

The generated data is operator-imitation data: the label is the selected filter's output, not an independently verified texture-free ground truth. Keep the source-image split fixed when comparing models, and report the exact operator and parameter distribution used to generate each training set.

`--spec-json` can override sampling ranges. For example:

```json
{
  "count": 7,
  "lambda_base": 1.0,
  "lambda_log_range": [-2.30258509299, 2.30258509299],
  "alpha_range": [1.0, 1.6]
}
```
