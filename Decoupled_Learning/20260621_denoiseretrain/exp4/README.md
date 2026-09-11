# Experiment 4A: Deep Parametric Cascade

This is the capacity-matched comparison for Experiment 3. It keeps the original
parameterized refine recurrence from Experiment 1:

`delta2 = a2 * delta1 + b2`, `delta3 = a3 * delta2 + b3`

Each of the three refine predictors uses five `3x3` convolutional layers,
matching Experiment 3's depth. ConvGuided stage 1 is frozen from `../exp1`.
BSDS400 whole-image training, geometry augmentation, BSD68 validation, and the
80-epoch schedule match Experiment 3. Sigma 25 is the only supported level.

Run on physical GPU 1:

```bash
bash run_exp4a_sigma25.sh 1
```

Artifacts are saved under `models/sigma25/` with the `exp4a_parametric_deep_`
prefix.

## Experiment 4B

Experiment 4B uses the identical five-layer parameterized model but replaces
Experiment 4A's delta supervision with Experiment 3's image supervision:
`0.2 * stage1 + 0.3 * stage2 + 0.5 * final`. This isolates the residual
parameterization while holding network capacity fixed.

```bash
bash run_exp4b_sigma25.sh 1
```
