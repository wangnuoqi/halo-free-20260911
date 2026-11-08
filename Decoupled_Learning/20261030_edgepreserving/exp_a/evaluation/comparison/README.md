# 20260818 测试集模型对比

测试集共 200 张图。新训练的 A1、A0、C1 使用 `final.pth`，全图推理，偶数边裁剪；历史 NAFNet、VIVNet、exp7C/C0 的指标读取自 `20260815_edgepreserving/baseline/evaluation/` 中已有的逐图记录。六组记录的图像 ID 去除文件名后缀后完全一致。

| 模型 | PSNR (dB) ↑ | MSE ↓ | 来源 / 说明 |
|---|---:|---:|---|
| NAFNet WLS | **49.434** | **1.476e-5** | 已有 NAFNet 全图测试，偶数边裁剪，BF16 |
| VIVNet WLS | 48.992 | 1.620e-5 | 已有 VIVNet 全图测试，补齐到 4 的倍数，FP32 |
| 旧 C0 / exp7C joint | 47.114 | 2.273e-5 | 已有 exp7C 全图测试，偶数边裁剪 |
| A0（旧错误输入训练） | 30.112 | 1.080e-3 | 修正推理流程重测；训练时 HR 重复减 0.5，结果无效 |
| A1（旧错误输入训练） | 7.749 | 1.861e-1 | 修正推理流程重测；训练时 HR 重复减 0.5，结果无效 |
| 旧 C1 scratch | 29.884 | 1.096e-3 | 原 C1，从头训练，DP，global batch 64，BF16；不代表只替换 LRNet |
| C1 exp7C transfer（旧错误输入训练） | 12.588 | 6.255e-2 | 200 epochs，DDP/NCCL，global batch 64；训练时 HR 输入重复减 0.5，结果无效 |

PSNR 是逐图 PSNR 的算术平均；MSE 是逐图 MSE 的算术平均，遵循各方法保存的逐图 CSV 汇总口径。六组都测了同一批图，但 VIVNet 的边界补齐方式与其他模型不完全相同，因此边界像素存在很小的协议差异。

## 本次 checkpoint

- A1: `../a1/summary.json`，`../a1/per_image_metrics.csv`
- A0: `../a0/summary.json`，`../a0/per_image_metrics.csv`
- C1: `../c1/summary.json`，`../c1/per_image_metrics.csv`
- A1/A0/C1 均为 epoch 200、6400 updates，保存学习率 `3e-6`。
- 本项目没有重新训练 C0；表中的 C0 是之前的 exp7C joint checkpoint 测试结果。
- 所有结果仅保存指标，没有保存预测图。

## C1 异常的核对结果

旧 exp7C 和新实现的 coarse 坐标流程相同：LR、HR 输入由调用方减 `0.5`，guided filter 的输出 clamp 到 `[0,1]`，再减 `0.5` 送入 refine。之前“新实现多做了一次 clamp”的判断不正确。

旧 C1、A0、A1 都没有加载旧 checkpoint。三个 checkpoint 的 `run_protocol.from_scratch` 为 `true`，都是从头初始化；旧 C1 不是旧 C0 上只替换 LRNet 的实验。

训练入口将 `lr` 和 HR `x` 中心化后传入 `Full.forward()`，但原实现又对 HR `x` 减了一次 `0.5`，造成 stage 看到错误的 HR 输入。OriginalStage 和 NAF Stage 的 Sobel 幅值也没有像旧 exp7C 那样按每张图归一化。A0、A1、C1 scratch 和首个 C1 transfer 都经过了这条错误数据流训练，不能用于模型结构对比。修正后将旧 LR、ConvGuided、stage checkpoint 全部映射到新实现，可在 200 张测试图上复现 `47.114087647 dB`，与 exp7C 记录一致，验证权重映射与评估流程。

因此 A0、A1、旧 C1 scratch 和首次 exp7C transfer 都不能用于判断结构改动是否有效。修正版 C1 迁移训练命令和输出目录见 `../README.md` 的 C1 部分。A0/A1 也必须用修正后的代码重新训练，才能和修正版 C1 及旧 exp7C 公平比较。
