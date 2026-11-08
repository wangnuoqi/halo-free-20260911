# exp1: coarse-edge gated 3-stage refine

This experiment is a direct alternative to the cached single-stage NAF refine. It uses the same cache manifest, `CachedRefineDataset`, spatial transforms, L1/LPIPS/gradient training loss, optimizer and cosine schedule as `../../refine_dataparallel.py`.

For stage `k`, the current output supplies the only structural edge map: `E_k = edge(C_k)`. `blended` does not supply an edge map or raw feature tensor to the residual predictor. It supplies only `HF(blended)-HF(C_k)`, which the existing in-stage gate filters before the predictor receives it. The low-frequency blended/current difference is retained for colour and illumination correction.

The stage gate follows the explicit prior/detail enhancement rule
`g = 1 + beta * P_int * P_hf` with `beta=0.25`, where `P_int` is the normalized
reflection prior and `P_hf` is normalized input high-frequency richness. It
enhances detail conditioning in locations that are both reflection-heavy and
detail-rich. The stage update remains `delta_k = delta_(k-1) + residual_k`, so
the three-stage residual backbone is unchanged.

Run four-GPU training from WSL:

```bash
bash train_wintrain_4gpu.sh
```

The trainer writes `checkpoint-latest.pth` after every epoch and at each
`--checkpointing_steps` interval. It contains the unwrapped model weights,
optimizer and scheduler states, epoch/step position, configuration, and RNG
states. Continue an interrupted run with the same command plus:

```text
--resume latest
```

The terminal progress bar reports the current loss, L1, LPIPS, and gradient
loss for each batch. Numbered checkpoints are retained according to
`--checkpoints_total_limit`; `refine_3stage_exp1_final.pth` is written at the
end of training.

Evaluate a saved checkpoint on a cache manifest with:

```bash
python evaluate.py --cache_root <cache-root> --cache_manifest <test-manifest.jsonl> --checkpoint <checkpoint.pth> --output_dir <metrics-dir>
```
