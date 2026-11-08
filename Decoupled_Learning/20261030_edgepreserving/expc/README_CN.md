# Exp C

Exp C tests structurally different, directly supervised single-channel models for the fixed 20260818 WLS task (`lambda=0.5`, `alpha=1.4`). See [IDEAS_CN.md](IDEAS_CN.md) for hypotheses, related work, controls, and the promotion gate.

## Models

| Variant | Main question | Parameters |
|---|---|---:|
| `naf_small` | Can a small two-scale NAF-style restoration model be a fast quality baseline? | 150,745 |
| `similarity_pyramid` | Does cross-scale feature similarity improve selective detail transfer? | 117,145 |
| `wls_naf` | Can a three-scale similarity-aware NAF encoder-decoder plus shallow edge correction act as a WLS-specific surrogate? | 547,043 |
| `lowres_edge` | Can a 1/4-scale base predictor plus gated high-frequency correction reduce cost? | 38,883 |
| `unrolled_diffusion` | Can a learned edge-conductance WLS-like solver explainably match the target? | 2,916 |
| `deep_guided_lr4` | Four VIV-inspired LR blocks followed by the learned affine guided filter. | measured at run time |
| `deep_guided_lr8` | Same model and protocol as `deep_guided_lr4`, with eight LR blocks. | measured at run time |
| `guided_pyramid_lr4` | Do four identical LR blocks work better when routed through a two-level encoder-decoder? | same as `deep_guided_lr4` |

All variants use the same Charbonnier + finite-difference loss, 0818 split, 128 crop, BF16, AdamW, and 60K optimizer steps. For parallel screening, four separate single-GPU processes use one card each and effective batch 64 by gradient accumulation; the formal WinTrain path uses four-GPU DataParallel. Training TIFF pairs are decoded once into a shared float32 NPY cache. Validation uses a fixed 200-image train holdout. Test-200 is only evaluated after selecting each best checkpoint.

The `deep_guided_lr4` / `deep_guided_lr8` pair uses the existing precomputed 2x LR TIFF, HR guide, and WLS target. Both candidates keep 24 channels, dilation cycle `[2, 4, 8, 16]`, block type, guided-filter coefficient head, loss, and training protocol fixed; only block count changes. Run both to the same 5K screening gate before deciding whether either deserves a 60K run. This tests depth at fixed width and resolution, not a U-Net or channel-width scaling.

`guided_pyramid_lr4` is the macro-architecture comparison to `deep_guided_lr4`: it preserves the same modules, parameter count, four block dilations and guided-filter head, but routes blocks through LR, LR/2 and LR/4 features with two skip fusions. Its first gate is coarse-only full-image validation at 5K updates; no Refine or joint stage is trained for this screening run. See [GUIDED_PYRAMID_CN.md](GUIDED_PYRAMID_CN.md).

## Four-GPU WinTrain run

Use the Windows CUDA runtime through `wintrain_logged.sh` and four-GPU DataParallel. The earlier four-process Gloo launcher failed during PyTorch rendezvous because the installed Windows build requests libuv although it was built without libuv. The DataParallel path avoids rendezvous. To train all candidates simultaneously, start one WinTrain process per candidate, each in its own tmux session and using all four visible GPUs. Existing `latest.pth` checkpoints resume automatically; `wls_naf` starts fresh if no checkpoint exists.

```bash
for variant in naf_small similarity_pyramid lowres_edge unrolled_diffusion wls_naf; do
  tmux new-session -d -s "expc_${variant}_4gpu" \
    "cd /mnt/d/ResearchCode/halo-free/Decoupled_Learning/20261030_edgepreserving/expc && WINTRAIN_LOG_PREFIX=expc_${variant}_4gpu ./run_expc_wintrain.sh --variant ${variant} --run-root runs/0818_memmap_4gpu_parallel_v2 --evaluation-root evaluation/test_20260818 --formal-iterations 60000 --screening-iterations 5000 --batch-per-gpu 16 --num-workers 4 --save-every 100 2>&1 | tee -a runs/${variant}_4gpu_tmux.log"
done
```

```bash
tmux new-session -d -s expc_0818 \
  'cd /mnt/d/ResearchCode/halo-free/Decoupled_Learning/20261030_edgepreserving/expc && ./run_expc_wintrain.sh 2>&1 | tee -a runs/wintrain_tmux.log'
tmux attach -t expc_0818
```

For a standalone `wls_naf` run only, the optional queue waits for all four GPUs
to remain below 10% utilization for three consecutive probes. Do not start it
while the all-candidate sequence above is running:

```bash
tmux new-session -d -s expc_wls_naf_0818 \
  'cd /mnt/d/ResearchCode/halo-free/Decoupled_Learning/20261030_edgepreserving/expc && ./run_wls_naf_queued.sh 2>&1 | tee -a runs/wls_naf_0818_tmux.log'
```

Artifacts go to `expc/runs/0818_four_hypotheses_4gpu/`; test summaries go to `expc/evaluation/test_20260818/<variant>/summary.json`. The PowerShell payload uses `conda run --no-capture-output -n yolo_clean` with unbuffered Python output.

`run_4gpu_queue.sh` remains as a native WSL/NCCL option for systems where CUDA is directly available. Do not run both launchers simultaneously because they share output paths.

## Screening protocol

The first pass uses the same 60K-step scheduler protocol but stops every candidate at
5K updates. This gives a common early comparison on the fixed train holdout without
touching `test-200`:

```bash
tmux new-session -d -s expc_0818_screen \
  'cd /mnt/d/ResearchCode/halo-free/Decoupled_Learning/20261030_edgepreserving/expc && ./run_expc_wintrain.sh --skip-smoke --screening-only --screening-iterations 5000 --num-workers 0 --run-root runs/0818_memmap_4gpu --evaluation-root evaluation/test_20260818 --formal-iterations 60000 --save-every 100 2>&1 | tee -a runs/0818_memmap_4gpu_screen_tmux.log'
```

Only after selecting a candidate from the train holdout should its checkpoint be
evaluated on the independent test set. Windows training uses pageable host transfers
to avoid unbounded pinned-memory growth in long DataParallel runs.

To screen the depth pair sequentially through WinTrain, use distinct run outputs
and keep the independent test set untouched:

```bash
tmux new-session -d -s expc_deepguided_depth \
  'cd /mnt/d/ResearchCode/halo-free/Decoupled_Learning/20261030_edgepreserving/expc && ./run_expc_wintrain.sh --skip-smoke --screening-only --variant deep_guided_lr4 --screening-iterations 5000 --num-workers 4 --run-root runs/deep_guided_lr_depth_screen --evaluation-root evaluation/deep_guided_lr_depth_screen --formal-iterations 60000 --save-every 100 2>&1 | tee -a runs/deep_guided_lr_depth_screen_tmux.log && ./run_expc_wintrain.sh --skip-smoke --screening-only --variant deep_guided_lr8 --screening-iterations 5000 --num-workers 4 --run-root runs/deep_guided_lr_depth_screen --evaluation-root evaluation/deep_guided_lr_depth_screen --formal-iterations 60000 --save-every 100 2>&1 | tee -a runs/deep_guided_lr_depth_screen_tmux.log'
```
