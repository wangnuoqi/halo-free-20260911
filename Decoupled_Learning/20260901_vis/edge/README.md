# Edge model comparison visualization

This directory is an isolated inference package. It does not modify training
experiments or generate offline training data.

## 1. Generate model outputs

From the `halofree` environment:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n halofree python \
  evaluation/run_inference_compare.py
```

The default test set is `S2R-HDR/data/mixed_lavel_singlehdr`. Passing
`--data-root /mnt/d/nuoqi/S2R-HDR/data/20260813` automatically resolves the
parameter-suffixed 200-image test directories. Outputs are
normalized float32 TIFFs under `outputs/mixed_lavel_singlehdr/`:

```text
baseline_original/  # original ConvGuided + original BatchNorm RF
exp7c_recipe/       # exp7C structure/recipe retrained on this label set
gt/                  # matched test labels
metrics.txt
```

The evaluator matches by leading numeric sample id, generates a missing
half-resolution input online, and uses each model's own ConvGuided checkpoint.

## 2. Reconstruct RGB comparison images in MATLAB

Open and run `visualization/reconstruct_color_compare.m` in MATLAB. Before
running, change `dataRoot` and `hdrRoot` only if the Windows data locations
differ. The script reads the original HDR image, restores detail using the
same logic as `test_tmcode_modelvis_params.m`, and saves:

```text
visualization/mixed_lavel_singlehdr_rgb/0001_compare.jpg
```

Each image contains `Original baseline | Exp7C recipe | WLS GT`. The legacy
`WLS-025000` label naming is interpreted as `lambda=0.25, alpha=1.2`; labels
with explicit `wls-l...-a...` names are parsed per sample.

Current output archive layout:

```text
outputs/
  test data/
    test_20260813/wls_l1p000_a1p600/
    test_20260818/wls_l0p500_a1p400/
    test_mixed_lavel_singlehdr/wls_l0p250_a1p200/
  test_fei/wls_l0p250_a1p200/       # reserved for the next test_fei run
  _archive_invalid/                 # previous invalid/incomplete runs
```
