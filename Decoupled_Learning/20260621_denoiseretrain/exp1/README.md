# Experiment 1: Original Full-Image Refine Baseline

This directory is a snapshot of the original full-image refinement baseline:

`noisy image -> frozen ConvGuided stage 1 -> original CascadeRefine3Stage`

`CascadeRefine3Stage` is the original refinement formulation. Its second and
third stages rescale and shift the previous residual; it is intentionally kept
unchanged here as the baseline for later experiments.

## Contents

- `train_rf_3stagev3_fullimage.py`: full-image BSDS400 training with BSD68
  validation.
- `eval_convguided_gray.py`: frozen ConvGuided stage-1 model and weight loader.
- `eval_rf_3stagev3.py`: original refine model, loss, and evaluation utility.
- `dncnn_gray_lr_patch_dataset.py`: stage-1 evaluation dataset dependency.
- `tests/`: full-image data-pipeline tests.

All original checkpoints, TensorBoard logs, and evaluation outputs now live in
this directory under `models/`, `log_dir/`, and `eval_results-img*/`. The
default experiment root is the directory containing each script, so new runs
remain self-contained in `exp1`.

`direct_residual_baseline/` contains the earlier direct-residual control
experiment and its artifacts, retained for comparison with the original
ConvGuided-plus-refine baseline.

Run from this directory:

```bash
python train_rf_3stagev3_fullimage.py --sigma 25 --device cuda:0
```

To explicitly select a different artifact root, pass `--experiment-root`.
