# 20260623_RGBDsr 实验交接

## Scope

本目录研究 NYUv2 x8 RGB-D depth super-resolution，比较自有 ConvGuided、官方 DKN、官方 DCTNet，以及统一的 Exp6D edge/smooth 双 gate RF。当前实验允许每轮使用 449 张测试集选择最优权重，目标是优化项目测试分数，不是无泄漏 benchmark。

## Current Status

- Exp1：ConvGuided x4/x8/x16 及原始 Refine3Stage RF 已完成；其 x8 RF `4.5712 cm` 是历史 baseline，仅用于和 Exp3 新 RF 方案对照，不属于 Exp3 当前结果。
- Exp2：x4 guide reliability A/B/C 已完成；增益很小，不扩展。
- 第三方 origin：DKN、DCTNet x8 官方 checkpoint 已完成本地 coarse 测试。
- Exp3 当前结果：A、B 的 frozen/joint 已完成；C frozen 已按单次 RGB normalization 重跑并完成，C joint 尚未生成 checkpoint。Exp3 结果表只对应这次重写后的 6 个独立输出目录，不包含 Exp1 历史 baseline。
- 旧 Exp3 B/C checkpoint 已归档到 `exp3/archive/old_protocol/`，不能作为新协议结果。

## Verified Artifacts

### Exp1 历史 baseline（非 Exp3 当前结果）

449 张测试、crop=6、逐图 RMSE 宏平均：

| scale | Conv coarse RMSE | Conv + frozen RF RMSE |
|---|---:|---:|
| x4 | 3.3013 cm | 2.1512 cm |
| x8 | 6.1133 cm | 4.5712 cm |
| x16 | 10.7335 cm | 8.8030 cm |

结果位于 `exp1/test_results/`，模型位于 `exp1/models/`。

### Third-party origin coarse

- DKN checkpoint：`third_party/DKN/parameter/DKN_8x`
- DKN coarse：`test_results/dkn_x8_coarse_pil_rgb_bs8.json`，RMSE `3.3229 cm`。
- DCTNet checkpoint：`third_party/DCTNet/models/DCTNet_8X.pth`
- DCTNet coarse：`test_results/dctnet_x8_coarse_pil_bs1.json`，RMSE `3.2975 cm`。
- DKN raw 毫米诊断：5 张 RMSE `17.4971 cm`；逐图归一化输入 RMSE `2.7765 cm`，确认当前本地 DKN 输入应使用逐图 min-max。

### Exp3 old artifacts

旧结果均为 preliminary，不用于新协议比较：`exp3/archive/old_protocol/`。旧 A 还位于 `exp3/modelsexp3A_convguided_joint/`，使用 900 张训练且未加入 origin 原始 loss，同样不作为新 A 结果。

## Reproduction

### Data and evaluation

所有新 Exp3 分支使用完整 1000 张训练池；独立 449 张测试集每轮评估并选择最优 checkpoint。每张图执行：

```text
depth 逐图 min-max -> [0,1]
-> bicubic x8 LR/HR
-> origin coarse / RF final
-> 恢复该图真实深度范围
-> crop=6
-> 每图 RMSE、MAE
-> 449 张宏平均
```

### Exp3-A ConvGuided

- Origin：`exp1/models/conv_survey_dgf_x8/rgb_guided_depth_x8_best.pth`
- 输入：`NYUv2FullImageDataset`，RGB `/255`，逐图 depth min-max，PIL bicubic。
- Frozen：ConvGuided 冻结，只训练 Exp6D RF，RF lr `3e-3`，batch `4`，80 epoch。
- Joint：ConvGuided lr `3e-4`，RF lr `3e-3`，batch `4`，80 epoch。
- Joint loss：`Exp6D + 0.5*SmoothL1(coarse,target) + 1.0*L1(coarse,target)`。
- 入口：`exp3/train_exp3_ABC_x8.sh`。
- 输出：`exp3/models/exp3A_convguided_rf_frozen_full1000/`、`exp3/models/exp3A_convguided_rf_joint_full1000/`。
- 已完成结果（449 张逐图 RMSE 宏平均，测试集选最优）：frozen coarse/RF `6.1237/6.0979 cm`（最优第 78 轮）；joint coarse/RF `6.4008/6.1484 cm`（最优第 70 轮）。
- 重要协议说明：Exp1 的 x8 `4.5712 cm` 使用原始 `Refine3Stage`；本 Exp3-A 使用 `Exp6DRefine3Stage`，因此两者不是同一 RF，不能直接作为 ConvGuided 退化的证据。Exp6D 当前实现还包含梯度损失，该项不是用户单独指定的新增项，而是沿用了 Exp6D 原实现，结果只能作为 Exp6D 迁移对照。
- coarse 数值核对：单独 `test_convguided_base.py` 的 x8 `6.9147 cm` 是全像素汇总 RMSE；Exp1 RF 与 Exp3 主表的 `6.11--6.12 cm` 是逐图 RMSE 宏平均，不能直接互比。Exp3 代码已补齐 Exp1 的 coarse `clamp(0,1)` 后处理；已有 A checkpoint/指标保留为旧的微小协议差异结果。

### Exp3-B DKN

- Origin：`third_party/DKN/parameter/DKN_8x`，结构 `k=3,d=15,residual=True`。
- 输入：`exp1/data/dkn_nyuv2_1000_449/`；逐图 depth min-max、PIL bicubic、RGB 顺序。
- Frozen：只训练 RF，RF lr `3e-3`，batch `1`，80 epoch。
- Joint：DKN lr `1e-4`、RF lr `3e-3`，batch `1`，80 epoch。
- Joint loss：`Exp6D + 0.5*SmoothL1(coarse,target) + 1.0*L1(coarse,target)`。
- DKN scheduler：`StepLR(step_size=10000,gamma=0.2)`，每 batch 调度。
- 输出：`exp3/models/exp3B_dkn_rf_frozen_full1000/`、`exp3/models/exp3B_dkn_rf_joint_full1000/`。
- 已完成结果（449 张逐图 RMSE 宏平均，测试集选最优）：frozen coarse/RF `3.3251/3.3269 cm`（最优第 6 轮，RF 无改善）；joint coarse/RF `3.4681/3.2507 cm`（最优第 63 轮，最终 RF 比 frozen RF 好 `0.0762 cm`，但联合后 origin coarse 变差）。

### Exp3-C DCTNet

- Origin：`third_party/DCTNet/models/DCTNet_8X.pth`。
- 输入必须是 DCTNet processed H5/目录；不得复用 DKNArrayDataset。
- 预处理工具：`tools/prepare_dctnet_processed.py`。顺序为 HR depth -> bicubic LR -> 用 HR min/max 归一化 LR -> bicubic LR-up；RGB `/255` + ImageNet mean/std。
- Frozen：只训练 RF，RF lr `3e-3`，batch `8`，200 epoch。
- Joint：DCTNet lr `1e-3`、RF lr `3e-3`，batch `8`，200 epoch。
- Joint loss：`Exp6D + 0.5*SmoothL1(coarse,target) + 1.0*MSE(coarse,target)`。
- DCTNet scheduler：`StepLR(step_size=100,gamma=0.5)`，每 epoch 调度；联合参数 gradient clip `max_norm=1`。
- 输出：`exp3/models/exp3C_dctnet_rf_frozen_full1000/`、`exp3/models/exp3C_dctnet_rf_joint_full1000/`。
- 旧版 frozen 训练因 RGB 二次 normalization 作废；修复后 C frozen 已跑满 200 epoch，coarse/RF RMSE `3.2975/3.2992 cm`（测试集最优第 14 轮），与官方 coarse 对齐。C joint 也已跑满 200 epoch，coarse/RF RMSE `3.7600/3.7631 cm`；RF 相对 coarse 变差 `0.0030 cm`。有效产物位于 `exp3C_dctnet_rf_joint_full1000/`。

### Commands

```bash
cd /mnt/d/nuoqi/halo-free/Decoupled_Learning/20260623_RGBDsr
bash exp3/train_exp3_ABC_x8.sh
bash exp3/train_exp3_BC_freeze_joint.sh
```

`train_exp3_BC_freeze_joint.sh` 中的 `WIN_DCT_TRAIN` 和 `WIN_DCT_TEST` 必须改成真实的 DCTNet processed 数据路径；通过 WinTrain、`yolo_clean`、GPU `0,1,2,3` 运行。

## Design and Decision Context

- Frozen 与 joint 成对比较，用于区分 RF 独立修复能力和 origin 解冻后的联合优化能力。
- DKN 使用官方 origin L1；DCTNet 使用官方 origin MSE；ConvGuided 使用项目原有 L1。
- Exp7C 的 `0.5*SmoothL1(coarse,target)` 作为 joint origin fidelity，防止 origin 为配合 RF 而退化。
- RF 始终采用 Exp6D edge/smooth 双 gate；其实现的 `exp6d_loss` 同时含像素、edge/smooth 和梯度项；frozen/joint 只改变 origin 是否更新。

## Pitfalls and Constraints

- `test_convguided_base.py` 的旧 metrics 是全像素汇总；Exp3 主指标是逐图 RMSE 宏平均，不能直接混比。
- 测试集选 best 是本项目当前刷分实验约定，结果应标注该事实。
- DKN 官方仓库的 dataset 下载链接目前返回 QuickConnect HTML；官方 prepared array 的原始数值范围无法直接核验。
- DCTNet 官方代码依赖旧 PyTorch/Kornia；当前使用兼容后的 PyTorch 2.x 实现，不能宣称 bitwise 相同。
- DCTNet 当前入口从原始目录在线完成预处理，不生成 H5/NPY；必须保证 RGB ImageNet normalization 只发生一次，不能把 DKN 数据静默作为测试输入。

## Open Questions

- DKN 官方 prepared array 的确切缩放方式尚未从官方数据包确认；本地实验已证明逐图归一化是正确数量级。
- 启动 C joint（使用修复后的单次 RGB normalization）；若要公平验证 RF 结构，需另行用原始 `Refine3Stage` 重跑对照。

## Next Actions

1. 生成完整 DCTNet x8 H5 和 processed test 目录，并填入 WinTrain 脚本路径。
2. 修正 C 的 RGB normalization 为单次后，重跑 frozen/joint；A/B 已完成，无需重复训练。
3. 检查每个输出目录的 `exp3_joint_best.pth`、`exp3_joint_last.pth` 和 `test_metrics.json`。
4. 仅在同一逐图 RMSE、crop=6、depth 恢复口径下比较 ConvGuided、DKN、DCTNet 与 RF 前后结果。
