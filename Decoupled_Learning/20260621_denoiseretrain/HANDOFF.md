# Exp7/Exp8/Exp9 Handoff

## Scope

This directory contains gray-scale sigma25 denoising experiments on BSDS400
and BSD68. The formal retrains are isolated under `exp7/formal_v2/` and
`exp8/formal_v2/`; Exp9 loss ablations are isolated under `exp9/formal_v1/`.
No formal script reads legacy `runs/` directories.

## Current Status

- **Verified:** Exp7 的 ConvGuided、DnCNN、FFDNet full-image base 均已完成，
  checkpoint 位于 `exp7/formal_v2/models/*_base_full/`。
- **Verified:** Exp7 的 DnCNN 联合/冻结，以及 FFDNet 联合 RF 分支均已完成，
  结果位于 `exp7/formal_v2/runs/` 对应目录。
- **Implemented, not run:** Exp7 的 ConvGuided 联合与冻结 RF 分支。
- **Incomplete:** `exp7B_ffdnet_frozen_exp6d_full/` 已有 checkpoint 和
  `metrics.json`，但缺少 `training_complete.json`，因此暂不视为完整正式结果。
- **Implemented, not run:** Exp8 patch-protocol six-arm matrix.
- **Implemented, not run:** Exp9 DnCNN/FFDNet joint patch loss ablation,
  consisting of four runs in `exp9/formal_v1/runs/`.
- **Preliminary / invalid for comparison:** all checkpoints under the old
  `exp7/runs/` and `exp8/runs/` paths. They predate the final noise and
  protocol corrections.
- **Verified:** `third_party/refine_runs/bsds400_bsd68_sigma25/` contains an
  earlier frozen-RF baseline, but it is not a replacement for formal_v2.

## Verified Artifacts

- `exp7/train_exp7.py`: shared train loop. Joint runs require Exp7C RF and
  the selectable coarse term; frozen runs require Exp6D.
- `exp7_8_common.py`: the default joint coarse term remains
  `0.5 * SmoothL1(coarse, clean)`, while Exp9 also exposes masked MSE and
  `MSE + 0.5 * SmoothL1` through `--coarse-loss`.
- `exp9/run_exp9.py`: runs the two Exp9 variants for one denoiser
  sequentially, stages its official sigma25 initializer, and removes only
  incomplete output directories before retrying.
- `exp7_8_common.py`: NumPy no-clamp sigma25 noise, Lanczos-4 ConvGuided
  input, dense DnCNN/FFDNet patch protocols, and RF implementations.
- `train_convguided_base.py`: fresh experiment-local ConvGuided base trainer.
- `evaluate_bundle.py`: evaluates a bundle using full BSD68 and its recorded
  RF type, rather than hard-coding legacy Exp6E.

## Exp7 当前测试结果

以下数值来自各 run 的 `metrics.json`，对应训练过程中对完整 BSD68 的固定测试：
测试噪声为每张图重新初始化的 NumPy `RandomState(0)` sigma25 AWGN，输出 clamp
到 `[0, 1]`。因此这是当前代码协议下的 BSD68 测试结果，不是训练集指标。

| 分支 | Denoiser | RF | 状态 | Coarse PSNR | Final PSNR | Final - Coarse |
|---|---|---|---|---:|---:|---:|
| Base | ConvGuided | 无 | 已完成 | 26.680143 | - | - |
| Base | DnCNN | 无 | 已完成 | 29.245222 | - | - |
| Base | FFDNet | 无 | 已完成 | 29.249341 | - | - |
| Exp7A | DnCNN | Exp7C | 已完成，best epoch 96 | 29.241579 | 29.245549 | +0.003970 |
| Exp7B | DnCNN | Exp6D | 已完成，best epoch 60 | 29.245224 | 29.245835 | +0.000611 |
| Exp7A | FFDNet | Exp7C | 已完成，best epoch 126 | 29.232723 | 29.246104 | +0.013381 |
| Exp7B | FFDNet | Exp6D | 未收尾，best epoch 2 | 29.249228 | 29.245338 | -0.003890 |

相对各自本地 base：DnCNN Exp7A 最终为 `+0.000327 dB`，Exp7B 为
`+0.000613 dB`；FFDNet Exp7A 为 `-0.003237 dB`。FFDNet Exp7B 的数值只作
阶段性记录，不能作为最终结论，直到补齐完整训练标记并重新确认。

Exp7 当前没有 ConvGuided RF 的正式测试结果；其 full-image base 已完成，后续
仍需运行 Exp7C 联合和 Exp6D 冻结两组，再统一比较三个模型的 coarse/final。

## Reproduction

For highest throughput, run:

```bash
bash run_formal_parallel_wintrain.sh
```

It schedules four independent one-GPU processes, each with its own
`CUDA_VISIBLE_DEVICES` value. This preserves the configured per-experiment
batch size and does not use DDP or DataParallel. Logs are written under each
formal root's `logs/` directory. The scheduler waits for each experiment-local
ConvGuided base before launching its two ConvGuided RF branches.

The launchers copy DnCNN/FFDNet initializers into the respective formal root,
then create all models and bundles there. A non-empty output directory fails
instead of being overwritten.

Exp9 明确使用单卡，并在同一张卡上串行运行该网络的两个消融。启动命令为：

```bash
GPU_ID=0 bash /mnt/d/nuoqi/halo-free/Decoupled_Learning/20260621_denoiseretrain/exp9/run_exp9_dncnn_single_gpu_wintrain.sh
GPU_ID=1 bash /mnt/d/nuoqi/halo-free/Decoupled_Learning/20260621_denoiseretrain/exp9/run_exp9_ffdnet_single_gpu_wintrain.sh
```

Each command uses `num-workers=8` and preserves the official patch protocol.

Exp7 的三个 full-image base 可以并行启动：

```bash
bash /mnt/d/nuoqi/halo-free/Decoupled_Learning/20260621_denoiseretrain/exp7/run_exp7_bases_parallel_wintrain.sh
```

该入口使用四个 GPU slot，分别调度 ConvGuided、DnCNN、FFDNet；由于只有
三个 base，第四张卡保持空闲。每个进程只看到自己的 GPU，不使用 DDP。

Exp7 base 完成后的 RF 启动入口为：

```bash
# 四卡 DataParallel：DnCNN 联合 Exp7C 后，再训练冻结 Exp6D
bash /mnt/d/nuoqi/halo-free/Decoupled_Learning/20260621_denoiseretrain/exp7/run_exp7_dncnn_rf_four_gpu_wintrain.sh

# 单卡 GPU0：FFDNet 联合 Exp7C 后，再训练冻结 Exp6D
bash /mnt/d/nuoqi/halo-free/Decoupled_Learning/20260621_denoiseretrain/exp7/run_exp7_ffdnet_rf_single_gpu_wintrain.sh

# 单卡 GPU0：ConvGuided 联合 Exp7C 后，再训练冻结 Exp6D
bash /mnt/d/nuoqi/halo-free/Decoupled_Learning/20260621_denoiseretrain/exp7/run_exp7_convguided_rf_single_gpu_wintrain.sh
```

DnCNN 使用四卡且 global batch 为 16（每卡约 4 张图）；FFDNet 和
ConvGuided 均使用 GPU1、batch 8。后两个命令都绑定 GPU1，因此不能并行。

如需一次串行完成全部六个 Exp7 RF 分支，使用：

```bash
bash /mnt/d/nuoqi/halo-free/Decoupled_Learning/20260621_denoiseretrain/exp7/run_exp7_all_rf_serial_wintrain.sh
```

顺序固定为 DnCNN 四卡两组、FFDNet GPU0 两组、ConvGuided GPU0 两组；任一
阶段失败时脚本立即停止，后续阶段不会启动。

## Design And Decision Context

| Experiment | Data protocol | Models | Training matrix |
|---|---|---|---|
| Exp7 | Full image, 8-way geometry, online sigma25 | ConvGuided, DnCNN, FFDNet | fine-tune a local full-image base per model, then joint Exp7C RF and frozen Exp6D |
| Exp8A | 128 HR / 64 LR, 32 patches/image | ConvGuided | joint Exp7C RF and frozen Exp6D |
| Exp8B | DnCNN 40x40, four scales, stride10 dense grid | DnCNN | joint Exp7C RF and frozen Exp6D |
| Exp8C | FFDNet 70x70, stride60 dense grid, random scale | FFDNet | joint Exp7C RF and frozen Exp6D |
| Exp9B | DnCNN official 40x40, four scales, stride10 dense grid | DnCNN | Exp7C joint RF with MSE, or MSE plus `0.5 * SmoothL1` |
| Exp9C | FFDNet official 70x70, stride60 dense grid, random scale | FFDNet | Exp7C joint RF with MSE, or MSE plus `0.5 * SmoothL1` |

Exp7 trains a local full-image MSE base for each of ConvGuided, DnCNN, and
FFDNet from its experiment-local initializer. Its full-image base and RF runs
use batch 8 by default,
160 epochs, RF lr `3e-3`, joint denoiser lr
`1e-4`, and decay at epochs 60/120. Exp8B follows DnCNN's 50-epoch frontend
schedule (`1e-3` then `1e-4`); Exp8C follows FFDNet's 500-epoch schedule
(`1e-4`, divide by 3 each 100 epochs, weight decay `5e-4`).

Exp9 answers whether the additional coarse SmoothL1 regularizer helps when
the original denoiser objective is restored during joint RF training. The
original DnCNN/FFDNet coarse objective is masked MSE, so the two variants are:

```text
MSE variant:        L = L_RF + MSE(coarse, clean)
MSE+Smooth variant: L = L_RF + MSE(coarse, clean) + 0.5*SmoothL1(coarse, clean)
```

Exp9B uses batch 128, 50 epochs, denoiser lr `1e-3`, weight decay `0`, and
RF lr `3e-3`. Exp9C uses batch 64, 500 epochs, denoiser lr `1e-4`, weight
decay `5e-4`, and RF lr `3e-3`. Both use `--mode joint`, `--rf exp7c`, the
same corrected online sigma25 noise/test protocol, and `num-workers=8`.
ConvGuided is excluded from Exp9 because this ablation targets the original
DnCNN/FFDNet coarse objectives; its custom coarse training is unchanged.

## Pitfalls And Constraints

- Training noise is random NumPy AWGN without noisy clamp. BSD68 test noise
  resets NumPy `RandomState(0)` per image; test outputs are clamped.
- FFDNet is fixed to sigma25 for this comparison, so it deliberately does not
  reproduce FFDNet's original random-sigma training target.
- Exp8B/C dense-grid data must not be replaced with the former random
  patches-per-image sampler.
- Exp9 must keep the two loss variants separate; do not interpret the default
  Exp7/8 `smooth` mode as the original DnCNN/FFDNet MSE objective.
- `20260621_denoiseretrain` is the correct directory spelling.

## Next Actions

如需加速单个 DnCNN full-image base，可使用四卡 DataParallel：

```bash
bash /mnt/d/nuoqi/halo-free/Decoupled_Learning/20260621_denoiseretrain/exp7/run_exp7_dncnn_base_four_gpu_wintrain.sh
```

该命令只运行 `dncnn-base`，把一个 batch 分到 GPU0-3；它与单卡脚本不同，
可能受到跨卡同步影响，应以实际 epoch 耗时判断是否加速。

1. Run Exp7 and Exp8 formal_v2 launchers.
2. Run the two Exp9 single-GPU launchers; each produces two sequential runs.
3. Run `evaluate_bundle.py` for every saved best bundle and store JSON under
   each experiment's `results/` directory.
4. Compare coarse and final PSNR only within the shared BSD68 test protocol,
   then report the Exp9 SmoothL1 ablation against its MSE-only control.
