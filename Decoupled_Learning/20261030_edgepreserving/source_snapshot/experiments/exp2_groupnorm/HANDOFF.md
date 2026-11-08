# exp2_groupnorm Handoff

## Scope

This experiment evaluates the ConvGuided + three-stage refine pipeline on the 20260818 WLS dataset. The purpose of exp2 is to test whether replacing BatchNorm with GroupNorm improves full-image refine training under small per-GPU batches.

## Current Status

**Verified complete.** The TensorBoard event file contains 80 `train/mse_epoch`, 80 `test/mse`, and 80 `test/psnr` values at steps 0--79. The original exp2 checkpoint is:

`experiments/exp2_groupnorm/models/exp2_groupnorm_rf_3stage_20260818_l0p500_a1p400/rf_3stage_best_20260818_l0p500_a1p400.pth`

The current best follow-up checkpoint is exp5B:

`experiments/exp5/exp5B/models/exp5B_groupnorm_lr3e3/rf_3stage_best_20260818_l0p500_a1p400.pth`

## Verified Artifacts

- Training source: `experiments/exp2_groupnorm/train_rf_exp2_groupnorm_20260818.py`
- Launcher: `experiments/exp2_groupnorm/run_exp2_groupnorm.ps1`
- Dataset root: `D:\nuoqi\S2R-HDR\data\20260818` on Windows, `/mnt/d/nuoqi/S2R-HDR/data/20260818` on WSL.
- Train split: `traininput-l0p500-a1p400` + `trainlabel-l0p500-a1p400`, 2003 pairs.
- Train low-resolution input: `traininput-l0p500-a1p400_downsample`.
- Test split: `testinput-l0p500-a1p400-200` + `testlabel-l0p500-a1p400-200`, 200 pairs.
- Test low-resolution input: `testinput-l0p500-a1p400-200_downsample`.
- Frozen first stage: `models/convguided_20260818_l0p500_a1p400/convguided_filter_best_20260818_l0p500_a1p400.pth`.
- Refine coarse images are generated online by the frozen ConvGuided model for every train/test batch.
- Evaluation report: `evaluation_compare_exp2_groupnorm/metrics.txt`.

## Reproduction

Run the existing four-GPU launcher through WinTrain:

```bash
bash run_refine_exp2_exp3.sh
```

The launcher checks the exp2 checkpoint and then proceeds to exp3. To run exp2 alone from Windows PowerShell, use `experiments/exp2_groupnorm/run_exp2_groupnorm.ps1` with `CUDA_VISIBLE_DEVICES=0,1,2,3` and the `yolo_clean` environment.

The common test command supports the experiment-specific architecture:

```bash
python evaluate_20260819.py \
  --rf-source experiments/exp2_groupnorm/train_rf_exp2_groupnorm_20260818.py \
  --rf-checkpoint experiments/exp2_groupnorm/models/exp2_groupnorm_rf_3stage_20260818_l0p500_a1p400/rf_3stage_best_20260818_l0p500_a1p400.pth
```

## Design and Decision Context

The original refine baseline uses BatchNorm in `Stage1DeltaNet` and both `ABAdjustNet` stages, plus the original dynamic-ratio edge loss. exp1 changed only the loss to explicit `smooth + edge + gradient` weighting and did not improve the baseline. exp2 is based on exp1 and changes the six refine normalization layers:

- `BatchNorm2d(16)` -> `GroupNorm(4, 16)`
- `BatchNorm2d(32)` -> `GroupNorm(8, 32)`

The model still freezes ConvGuided, generates coarse output online, uses 80 epochs and the original optimizer/scheduler, and selects the best checkpoint using test-set MSE as requested for this project.

## Results

All results below use the same 200-image test split, the same frozen ConvGuided checkpoint, and `evaluate_20260819.py`:

| System | MSE | PSNR |
| --- | ---: | ---: |
| ConvGuided only | 0.00014674 | 38.8956 dB |
| Original refine baseline | 0.00007666 | 41.7991 dB |
| exp1 explicit loss | 0.00007871 | 41.6509 dB |
| exp2 GroupNorm | **0.00005978** | **42.8168 dB** |

### Follow-up experiments

The following experiments use the same 20260818 test-200 split, frozen ConvGuided checkpoint, online coarse generation, and `evaluate_20260819.py` protocol. Their architecture-specific source is passed with `--rf-source` when evaluating GroupNorm or the exp4/exp5 variants.

| System | Main change | MSE | PSNR |
| --- | --- | ---: | ---: |
| exp3 soft-edge | Continuous normalized GT-gradient edge mask | 0.00008634 | 41.2174 dB |
| exp4A | GroupNorm-only ablation with original dynamic-ratio loss | 0.00005666 | 43.0411 dB |
| exp4B | GroupNorm + stage weights `0.1/0.3/0.6` | 0.00005507 | 43.1911 dB |
| exp4C | exp4B + final SmoothL1 | 0.00005894 | 42.8551 dB |
| exp4D | exp4C + final Sobel gradient loss | 0.00005848 | 42.9000 dB |
| exp4E | exp4D + smooth-region residual penalty | 0.00005677 | 43.0659 dB |
| exp5A | Stage weights `0.05/0.25/0.70` | 0.00005859 | 42.8753 dB |
| exp5B | exp4B with learning rate `3e-3` | **0.00005233** | **43.3998 dB** |
| exp5C | exp4B with learning rate `1e-3` | 0.00005360 | 43.2862 dB |
| exp5D | exp4B + identity initialization | 0.00005555 | 43.1370 dB |
| exp5E | exp4B + Stage3 residual damping `0.5` | 0.00005446 | 43.2124 dB |

Relative to the original refine baseline, exp5B reduces MSE by approximately 31.74% and improves PSNR by 1.6007 dB. Relative to exp4B, it reduces MSE by approximately 4.98% and improves PSNR by 0.2087 dB. The current ranking is `exp5B > exp5C > exp5E > exp5D > exp5A`.

Evaluation reports are stored in `evaluation_exp4_exp4A/` through `evaluation_exp4_exp4E/` and `evaluation_exp5_exp5A/` through `evaluation_exp5_exp5E/`, with the numeric summary in each `metrics.txt`.

## exp6 structural experiments

All exp6 variants were trained for 80 epochs. They inherit exp5B's GroupNorm, explicit edge/smooth/gradient loss, stage weights `0.1/0.3/0.6`, online coarse generation, and `lr=3e-3` unless noted:

| Experiment | Structural change | Output |
| --- | --- | --- |
| exp6A | Learnable per-pixel gate at all three stages | `experiments/exp6/exp6A/` |
| exp6B | exp6A + bounded residual magnitude (`0.15*tanh(update/0.15)`) | `experiments/exp6/exp6B/` |
| exp6C | Shared Stage2/Stage3 refinement cell | `experiments/exp6/exp6C/` |
| exp6D | Separate edge and smooth gate branches | `experiments/exp6/exp6D/` |
| exp6E | exp6A + joint ConvGuided/refine fine-tuning (`3e-4` for ConvGuided) | `experiments/exp6/exp6E/` |

The serial four-GPU launcher is `run_exp6_ABCDE.sh`. exp6E additionally saves `convguided_joint_best_20260818_l0p500_a1p400.pth`; its evaluation passes that checkpoint with the evaluator's `--conv-checkpoint` option.

### exp6 evaluation

All five experiments completed 80 epochs and were evaluated on the same 200-image test split. For exp6A-D, the ConvGuided checkpoint is the original frozen checkpoint. For exp6E, both ConvGuided and refine use the jointly trained checkpoints.

| System | ConvGuided MSE | RF final MSE | RF final PSNR |
| --- | ---: | ---: | ---: |
| exp6A learnable gate | 0.00014674 | 0.00005213 | 43.4484 dB |
| exp6B bounded residual | 0.00014674 | 0.00005381 | 43.3340 dB |
| exp6C shared refine cell | 0.00014674 | 0.00005355 | 43.2979 dB |
| exp6D edge/smooth gate | 0.00014674 | **0.00005193** | **43.4593 dB** |
| exp6E joint ConvGuided + refine | 0.00457131 | **0.00002613** | **46.4830 dB** |

Relative to exp5B (`0.00005233` / `43.3998 dB`), exp6A gives a small improvement, exp6B and exp6C regress, and exp6D gives a small improvement. exp6E is substantially better in final RF output, but its ConvGuided-only output is much worse (`0.00457131` MSE); the refine module is compensating for the jointly fine-tuned first stage. This result should therefore be treated as a different jointly optimized pipeline, not as a pure refine-only gain.

Evaluation reports are in `evaluation_exp6_exp6A/` through `evaluation_exp6_exp6E/`, each with a numeric `metrics.txt`.

## exp7 cross-dataset and ConvGuided-fidelity experiments

All exp7 experiments are now trained. `exp7A` applies the exp6E joint ConvGuided + gated refine protocol to the complete 20260813 WLS dataset (`l1p000/a1p600`) using its matching ConvGuided checkpoint. `exp7B` applies the same protocol to `mixed_lavel_singlehdr`; its test evaluation uses all 140 `testinput + testlabel` samples and generates the missing `0140` downsample online. `exp7C` keeps the 20260818 data and adds `0.5 * SmoothL1(ConvGuided coarse, GT)` to prevent the ConvGuided stage from collapsing during joint training.

The serial launcher is `run_exp7_ABC.sh`. Outputs are isolated under `experiments/exp7/exp7A_20260813_joint`, `experiments/exp7/exp7B_mixed_joint`, and `experiments/exp7/exp7C_20260818_joint_fidelity`.

exp7C completed 80/80 epochs. Its lowest training-time test MSE was `0.0000227272` at step 77 (epoch 78), with test PSNR `47.1141 dB`; the last epoch was `0.0000251516` / `46.6635 dB`.

Independent evaluation on the fixed 20260818 test-200 using the exp7C joint ConvGuided checkpoint gives ConvGuided MSE `0.00044515` / PSNR `33.5392 dB` and RF final MSE `0.00002273` / PSNR `47.1141 dB`. Compared with exp6E (`0.00457131` ConvGuided MSE, `0.00002613` RF final MSE, `46.4830 dB`), exp7C reduces ConvGuided MSE by about `90.26%`, reduces final RF MSE by about `13.01%`, and improves final PSNR by `0.6311 dB`. The independent report is `evaluation_exp7_exp7C/metrics.txt`.

### exp7A/exp7B test results

Both experiments were evaluated with their jointly trained ConvGuided and RF checkpoints using `evaluate_exp7_ab.py` on the matching test triples.

| Experiment | Matched test triples | ConvGuided MSE / PSNR | RF final MSE / PSNR |
| --- | ---: | ---: | ---: |
| exp7A, 20260813 | 200 | 0.16369316 / 7.8597 dB | 0.00005236 / 42.8100 dB |
| exp7B, mixed_lavel_singlehdr (早期 139 交集结果) | 139 | 0.15603655 / 8.0677 dB | 0.00000992 / 50.0370 dB |
| exp7B, mixed_lavel_singlehdr (最终 140 样本) | 140 | 0.15603833 / 8.0677 dB | **0.00000996 / 50.0180 dB** |

The ConvGuided-only values are severely degraded because these are joint-training checkpoints: the RF stage is compensating for the degraded coarse output. Therefore the RF final values describe the complete joint pipeline, while they are not evidence that the ConvGuided stage itself improved. The final exp7B evaluation uses all 140 `testinput + testlabel` samples; the missing `0140` downsample is generated online.

exp7A 的 label 是 WLS `lambda=1.0`、`alpha=1.6`，文件命名为 `WLS-l1p000-a1p600`。exp7B 的最终测试结果以 140 个 `testinput + testlabel` 样本为准；缺失的 `0140` 下采样输入按同一 Lanczos 半长边下采样规则在线生成。报告分别为 `evaluation_exp7_exp7A/metrics.txt` 和 `evaluation_exp7_exp7B_all140/metrics.txt`。

Compared with the original refine baseline, exp2 reduces MSE by approximately 22.0% and improves PSNR by approximately 1.02 dB.

## Pitfalls and Constraints

## exp8 cross-dataset fidelity retraining

exp8 reuses the exp7C joint-training recipe on the two non-20260818 datasets. Both variants use GroupNorm gated three-stage Refine, joint ConvGuided/Refine optimization, refine learning rate `3e-3`, ConvGuided learning rate `3e-4`, stage weights `0.1/0.3/0.6`, and the coarse fidelity term `0.5 * SmoothL1(coarse, GT)`. The coarse output is generated online and the first-stage checkpoint is fine-tuned rather than frozen.

- `exp8A`: 20260813 WLS (`lambda=1.0`, `alpha=1.6`), initialized from `models/convguided_20260813_l1p000_a1p600/convguided_filter_best_mix.pth`; source `experiments/exp8/exp8A_20260813_joint_fidelity.py`.
- `exp8B`: `mixed_lavel_singlehdr`, initialized from `/mnt/d/nuoqi/halo-free/Decoupled_Learning/total-pipeline-v1/checkpoints/convguided_filter_best_mix.pth`; source `experiments/exp8/exp8B_mixed_joint_fidelity.py`.
- Serial four-GPU launcher: `run_exp8_AB.sh` (WinTrain/yolo_clean, GPUs `0,1,2,3`).
- Checkpoints are isolated under `experiments/exp8/exp8A_20260813_joint_fidelity/` and `experiments/exp8/exp8B_mixed_joint_fidelity/`.
- Both experiments completed 80 epochs. The reported values below are their training-time test-set model-selection metrics, not post-training independent evaluations.

| Experiment | Dataset / test samples used in training | Best test step | Best test MSE | Best test PSNR | Final-epoch MSE / PSNR |
| --- | --- | ---: | ---: | ---: | ---: |
| exp8A | 20260813 WLS (`lambda=1.0`, `alpha=1.6`), 200 | 78 | 0.0000513104 | 43.6598 dB | 0.0000529665 / 43.4979 dB |
| exp8B | `mixed_lavel_singlehdr`, 139 matched triples | 77 | 0.0000060199 | 52.6665 dB | 0.0000065146 / 52.3259 dB |

The checkpoints selected by these metrics are:

- exp8A RF: `experiments/exp8/exp8A_20260813_joint_fidelity/models/exp8A_20260813_joint_fidelity/rf_3stage_best_20260818_l0p500_a1p400.pth`
- exp8A ConvGuided: `experiments/exp8/exp8A_20260813_joint_fidelity/models/exp8A_20260813_joint_fidelity/convguided_joint_best_20260818_l0p500_a1p400.pth`
- exp8B RF: `experiments/exp8/exp8B_mixed_joint_fidelity/models/exp8B_mixed_joint_fidelity/rf_3stage_best_20260818_l0p500_a1p400.pth`
- exp8B ConvGuided: `experiments/exp8/exp8B_mixed_joint_fidelity/models/exp8B_mixed_joint_fidelity/convguided_joint_best_20260818_l0p500_a1p400.pth`

No `evaluation_exp8A` or `evaluation_exp8B` report exists yet. In particular, exp8B's training test set is the 139-file intersection; its final report should follow the exp7B all-140 protocol, generating missing low-resolution input online where necessary.

- exp2 is not a pure GroupNorm-only comparison against the original baseline because it also inherits exp1's explicit loss. exp1's standalone result was worse than baseline, so GroupNorm is the leading explanation for the gain, but the loss/normalization interaction is not isolated.
- GroupNorm is required when loading this checkpoint; the generic evaluator must be passed `--rf-source experiments/exp2_groupnorm/train_rf_exp2_groupnorm_20260818.py`.
- Test-set model selection is intentionally retained and is therefore not a leakage-free validation protocol.
- Do not compare against the separate DualCNN + RF metrics; those use a different first-stage model and evaluation pipeline.

## Open Questions

- Does GroupNorm alone with the original loss outperform baseline?
- Does GroupNorm plus the original loss avoid the exp1 degradation?
- Are the gains retained under full-frame evaluation and edge/halo-specific metrics?

## Next Actions

1. Keep exp5B as the current best 20260818 ConvGuided + refine checkpoint and use its `3e-3` learning rate as the next baseline.
2. Confirm exp5B's gain with an independent validation/test split or multiple random seeds; current model selection still uses test-set MSE.
3. For further architecture work, compare edge/flat-region metrics in addition to full-image MSE and PSNR.
