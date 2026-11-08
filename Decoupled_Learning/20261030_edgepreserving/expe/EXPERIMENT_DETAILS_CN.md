# Exp E 实验细节与对照记录

本文档集中记录原 Exp7C、当前 E1、以及按当前协议重训的 E0/E1 对照。目的是区分历史系统结果与可归因的同协议实验。指标只在标注的 checkpoint、数据 split 和输入几何范围内有效。

## 状态标签

- **已验证**：配置和对应产物均存在，或数值已经由记录文件/评估脚本核验。
- **历史参考**：权重和指标存在，但训练协议、checkpoint 选择方式或输入几何与当前方案不同，不能单独用于架构归因。
- **已实现，未完成训练**：模型和训练入口可用，但尚无足够 checkpoint/验证结果。
- **未核实**：仓库中没有可靠的运行记录；不以代码默认值推断实际训练过程。

## 对照概览

| 名称 | LR predictor | Guided filter | 当前可用结果 | 证据等级 |
|---|---|---|---|---|
| 原 Exp7C Joint | 原 24 通道、4 个 dilation block（2/4/8/16） | 原 3×3 ConvGuided | test-200：coarse 33.5392 dB，final 47.1141 dB | 历史参考 |
| E1 Base/Refine/Joint | VIVNet 三尺度 U-Net，宽度32，每尺度1个 block，无末端 refinement blocks | Base E1 使用原 ConvGuided | test-200：Base 47.1023 dB；Refine final 46.7654 dB；Joint coarse 47.0932 dB、final 46.5200 dB | 已验证，但训练协议与原 Exp7C 不同 |
| E0 旧运行 `seed42_rerun` | 复现原 Exp7C LRNet 结构 | 原 3×3 ConvGuided | validation：40.7244 dB，MSE `8.4637e-5` | 历史参考；实际双卡暴露预算未与 E1 四卡对齐 |
| E0/E1 旧结果差值 | 只改变 LR predictor 架构 | 使用相同原 ConvGuided | 旧 E0 40.7244 dB；E1 44.9715 dB；差4.2471 dB | 不能作为严格 Base 架构消融 |

### E2：旧 Exp7C 优化配方下的 patch 结构对照（已实现，待运行）

E2 使用 `expe/train_legacy_matched.py`，在旧 Exp7C 的**优化配方**下比较 E0/E1 coarse 架构，并按 `base -> refine -> joint` 连续初始化。两者都使用 128 HR/64 LR 随机对齐 patch；旧 Exp7C 的 512×512 全图训练不属于 E2 的输入几何。`legacy_matched_e0` 使用原 24 通道 dilation LRNet，`legacy_matched_e1` 使用 E1 VIVNet U-Net；二者共享 ConvGuided、ThreeStageRefine、数据划分和 loss。新匹配预算为双卡每卡 batch 16、累积 2、有效 batch 64、60K 更新，每阶段 384 万样本呈现。旧配方仍用 Adam，Base LR `1e-4`、Refine LR `1e-2`、Joint coarse/refine LR `3e-4/3e-3`；阶梯衰减按原 30/80 和 60/80 的进度比例放在第 22,500/45,000 步。best 只按内部 full-image validation 选择，不用 test 选模。

因此，E2 是**固定 128/64 patch 输入后，比较旧/新网络在旧优化配方下表现**，不是原 Exp7C 的完整复刻。它控制 E0/E1 两个网络之间的训练设置，但不能把 E2 结果直接当作原 Exp7C 全图训练结果，也不能仅凭它解释全图历史运行与 patch 运行的差异。

历史原 Exp7C 使用 2003 对、四卡 DataParallel、global batch4、80 epoch，约40,000次更新、160,000次样本呈现。新的 E2 留出200对验证，1803对训练，以有效 batch64 跑60K更新；它是固定暴露预算下的旧优化配方，不是历史训练的逐项复刻。

**2×2 统一预算**：以已完成的 E1 新配方 Base 为基准，四格均使用相同 train/validation ID、128/64 crop、有效 batch64、60K optimizer updates 和384万样本呈现。其余三格重新训练；原 E0 `seed42_rerun` 虽保存了60K步及最终协议，却有双卡运行的实际记录，不能再标为已对齐。旧配方与新配方的优化器、学习率、scheduler 和 loss 仍然不同，这些变化作为整套训练配方因素，不单独归因给某个超参数。

严格的 2×2 结果比较四个 Base coarse 输出。当前 WSL 顺序训练另外三格时，E0 新配方还训练 Refine/Joint（各200 epochs，双卡每 rank batch32），E0 旧配方也训练 Refine/Joint（各60K更新、有效 batch64）；E1 旧配方只训练 Base。现有 E1 新配方 Refine/Joint 为200 epochs；旧配方的阶段预算不同，因此下游结果不能直接称为逐阶段匹配的2×2。入口为 `expe/scripts/run_e_2x2_ddp_2gpu.sh`，使用 NCCL DDP、GPU 0/1、每 rank 8 workers；四格通过协议校验后，仅旧 `legacy_matched_patch128*` 目录归档到 `expe/archive/legacy_e_checkpoints/`。旧 E0 新配方但暴露量不匹配的 Base/Refine/Joint 和已完成的四卡 E1 三阶段保留在原位置，供历史对照。

双卡以每卡 micro-batch16、梯度累积2实现有效 batch64；四卡 E1 为每卡16、无累积。两者更新次数和样本呈现量一致，随机 crop 序列、设备并行方式和每设备统计仍可能不同。

**协议纠正记录（2026-11-07）**：v2 的 Base/Refine 学习率及 loss 配置错误，v3 不再使用；v4 虽修复阶段配置，但预算未与 E1 对齐。当前匹配预算的 checkpoint implementation_revision 为5，新完整链从 `expe/runs/legacy_matched_exposure3840000_v5/` 开始，不能续用旧目录。

旧 E2 v4 配置是双卡每 rank batch2、global batch4、80 epoch，预算与 E1 基准不匹配，保留作历史协议记录。v5 改为双卡每 rank micro-batch16、梯度累积2、有效 batch64、60K更新。旧 Exp7C 的全图训练仍与 v5 的128/64随机patch不同。

### E 系列优化配方 × LRNet 结构状态

| 优化配方 | 旧网络：Exp7C dilation LRNet + 原 ConvGuided | 新网络：VIVNet U-Net LRNet + 原 ConvGuided |
|---|---|---|
| **旧优化配方**：Adam；Base `1e-4`、Refine `1e-2`、Joint coarse/refine `3e-4/3e-3`；第22.5K/45K步衰减；有效 batch64、60K步 | E2 v5 旧网络待训练；历史 Exp7C 全图 checkpoint 只作参考 | E2 v5 新网络待训练 |
| **新优化配方**：AdamW；Base `3e-4→1e-6` cosine；有效 batch64、60K步 | E0 双卡匹配预算待重训；原 `seed42_rerun` 和下游权重为未对齐历史结果 | E1 四卡 Base 已完成，是统一预算基准；Refine/Joint 也有独立历史结果 |

这张表是新的训练计划与当前状态，不是已完成的四格结果。除 E1 新配方外，三格必须在独立目录从头训练，之后才能做 2×2 比较。

### A 系列与 E 系列的卡数和暴露量核对

| 运行组 | GPU 数 | 每卡 batch / global batch | 输入与预算 | 样本呈现量 | 状态与可比范围 |
|---|---:|---:|---|---:|---|
| 归档旧 A1/A2（固定均值窗口） | 2 | 16 / 32 | 128/64 patch，60K updates | 1.92M | 已完成，但与 E1/E0 的 global batch64 不对齐；只能作旧结果参考 |
| 新 A 系列（stats/direct/fixed boxes）双卡串行脚本 | 2 | micro 16 / effective 64（累积2） | 128/64 patch，60K updates | 3.84M | 五个变体均按该协议训练；含固定窗口新协议重训；world size 不同 |
| E1 新配方 | 4 | 16 / 64 | 128/64 patch，60K updates | 3.84M | 已完成；统一预算基准 |
| E0 新配方匹配重训 | 2 | micro 16 / effective 64（累积2） | 128/64 patch，60K updates | 3.84M | 待运行；旧 E0 权重只作历史参考 |
| E2 旧优化配方 patch 对照 | 2 | micro 16 / effective 64（累积2） | 128/64 patch，60K updates | 3.84M | E0/E1 两格待运行 |

因此，归档的旧固定窗口 A1/A2 是双卡、global32；当前 learned-kernel A 系列和 E2 v5 均使用双卡每卡micro-batch16、梯度累积2，使有效 global batch 达到64。world size 不同仍会改变批次组成，故不是逐步完全相同的训练轨迹。不要把归档 A 的 60K/global32 结果直接与 E1 或新 A 的 global64 结果当作严格单变量消融。

### E2 与原 Exp7C 的参数对齐边界

原 Exp7C 可从源码直接核实的是 Joint 阶段：Adam；coarse/LRNet 学习率 `3e-4`、Refine 学习率 `3e-3`；StepLR 每30 epoch乘0.5；80 epochs；Joint 级联损失加 `0.5 × SmoothL1(coarse, GT)`。E2 Joint 阶段复用了这些可见设置。E2 的独立 Base `1e-4`、Refine `1e-2` 阶段属于新建的三阶段配对训练链，并非原 Exp7C Joint 脚本本身的阶段；原脚本加载已有 ConvGuided checkpoint 后直接进行 Joint。

所以 E2 v5 只能称为“旧优化超参数在 E1 暴露预算下的 128 patch 配对训练”，不能称为原 Exp7C 的完整复刻。它使用 1803/200 的 train/validation 划分、双卡每卡 micro-batch16、累积2、60K更新；原 Exp7C 则使用512全图、global batch4、约40K更新，还按 test MSE 选 checkpoint。E2 使用内部 validation 选模。

## 原 Exp7C：历史运行

### 模型与参数量

旧模型实现保存在 `source_snapshot/experiments/exp7C_joint_fidelity_20260818.py`，原始脚本位于 `../20260815_edgepreserving/experiments/exp7C_joint_fidelity_20260818.py`。

| 模块 | 结构 | 参数量 |
|---|---|---:|
| LRNet | 1→24 stem；4个24通道 3×3 dilation block，dilation=2、4、8、16；24通道 tail 和1通道输出 | 26,461 |
| ConvGuided | 可学习3×3 box；2→32→32→1 的 1×1 系数网络 | 1,261 |
| coarse 合计 | LRNet + ConvGuided | 27,722（0.0277M） |
| 三阶段 Refine | 原三阶段残差/仿射更新网络 | 43,800（0.0438M） |
| coarse + Refine | 两个部分合计 | 71,522（0.0715M） |

参数量来自当前 Exp7C 结构的实际实例化统计；不表示训练期间激活量或运行速度。

### 已知 Joint 配置

依据旧训练脚本和当前数据文件抽查：训练数据来自 20260818 的预生成 LR 输入、HR input 和 WLS label；训练目录有 2003 对。抽查 HR input 与 WLS label 为 512×512，配对 LR 为 256×256；旧脚本按数据集完整尺寸前向，仅将奇数边界裁成偶数，不做 128×128 随机 crop。batch size 随可见 GPU 数设置，四卡时 global batch 为4，使用 `DataParallel`（每卡约1张）。Joint 阶段 Refine 使用 Adam 学习率 `3e-3`、coarse/Base 使用 `3e-4`；StepLR 每30 epoch乘0.5，共80 epochs。Joint loss 为原三阶段边缘引导级联 loss，再加 `0.5 × SmoothL1(coarse, GT)`。旧训练 loader 对 2003 对数据使用 batch4、`drop_last=True`，约500 optimizer updates/epoch、80 epoch 约40,000 updates。旧 checkpoint 按 test MSE 选取，存在 test 集参与选模的风险。

旧脚本在 test 数据上逐 epoch 评估并按 test MSE 保存 best。因此旧 checkpoint 的 test 指标存在 test 集参与选择的风险。当前仓库没有旧完整运行配置快照或足够的 optimizer 状态可复原全部细节；上面未列出的超参数标为**未核实**。

### 旧 checkpoint 的当前 test-200 结果

结果来源：`evaluation/test_200_512/summary.json` 和 `evaluation/test_200_512/per_image_metrics.csv`。评估使用 FP32 单卡、同一20260818 test-200；HR/GT为中心512×512区域，LR为严格对齐的256×256区域。

| 输出 | Mean per-image PSNR | Mean image MSE |
|---|---:|---:|
| 原 Exp7C Joint coarse | 33.5392 dB | 4.45145e-4 |
| 原 Exp7C Joint final | 47.1141 dB | 2.27272e-5 |

Refine 将 coarse 拉升约13.57 dB，说明它在这组历史 checkpoint 中承担了很大幅度的误差补偿。但这不证明 coarse 学习率单独造成了退化：Base、Refine、数据几何、batch、优化器、scheduler 和训练预算均与当前 E1 不同。

## E1：当前已训练结果

E1 使用 VIVNet U-Net 预测半分辨率 `y_lr`，后接原 ConvGuided。训练输入仍是配对的半分辨率 LR TIFF；HR guide 是原图。模型参数统计如下：

| 模块 | 参数量 |
|---|---:|
| VIVNet LR predictor | 902,426 |
| 原 ConvGuided | 1,261 |
| E1 coarse 合计 | 903,687（0.9037M） |
| 三阶段 Refine | 43,800（0.0438M） |
| E1 coarse + Refine | 947,487（0.9475M） |

参数量比旧 Exp7C coarse 大约32.6倍；两者的 ConvGuided 参数相同。该比例不能直接推出相同比例的前向耗时，因为卷积所在尺度、通道宽度和算子执行方式不同。

Base checkpoint：`runs/scratch/viv_gf_e1/seed42_ddpfix/best.pth`（最佳验证点55,000步；`latest.pth`/`final.pth`为60,000步）。Base 配置在对应 `protocol.json`：四卡 DDP/NCCL、BF16、每卡 batch16（global64）、128×128 HR随机 crop/64×64 LR crop、AdamW、学习率 `3e-4` 到 `1e-6` cosine、60,000 optimizer steps、weight decay `1e-4`、MSE加动态平衡 Sobel 向量 L1。train 1803对，validation 200对，seed42。

外接三阶段 Refine/Joint 的结果同样见 `evaluation/test_200_512/README_CN.md`：

| 输出 | Mean per-image PSNR | Mean image MSE |
|---|---:|---:|
| E1 Base coarse | 47.1023 dB | 2.27596e-5 |
| E1 + frozen-Base Refine final | 46.7654 dB | 2.36654e-5 |
| E1 Joint coarse | 47.0932 dB | 2.28519e-5 |
| E1 Joint final | 46.5200 dB | 2.56660e-5 |

Refine 阶段冻结整个 Base，仅训练 Refine，AdamW，初始 LR `3e-4`、cosine 200 epochs 到 `1e-6`。Joint 从 E1 Base 和 Refine best 初始化，解冻两者，当前实现对全模型使用共享 LR `1e-5` 到 `1e-6`、cosine 200 epochs。E1 Joint coarse 与 Base 几乎一致；当前 test-200 上 Refine/Joint final 低于 E1 Base。这个观察说明当前 Refine 训练结果没有改善 E1，但不能据此判定 Refine 架构本身无效。

## 原 Exp7C 与当前 E1：对齐情况

### 已对齐

| 对比项 | 原 Exp7C | 当前 E1 | 判断 |
|---|---|---|---|
| 数据任务 | 20260818 WLS | 20260818 WLS | 对齐 |
| Guided filter | 原 ConvGuided | 原 ConvGuided | 对齐 |
| 最终 test 几何 | HR中心512、LR配对256 | HR中心512、LR配对256 | 对齐 |

### 未对齐

| 对比项 | 原 Exp7C | 当前 E1 | 影响 |
|---|---|---|---|
| LR predictor | 旧 dilation LRNet | VIVNet U-Net | 目标架构差异 |
| 训练输入几何 | 数据文件抽查为 HR 512×512 / LR 256×256 完整图，奇数边界裁切 | HR随机128 crop、LR配对64 crop | 不对齐；上下文范围和每步参与计算的像素数不同 |
| batch / 并行 | 四卡 DataParallel，global batch 4 | 四卡 DDP，global batch 64 | batch及梯度统计不同 |
| optimizer | Adam | AdamW，weight decay `1e-4` | 更新规则不同 |
| Base 学习率 | Joint中 `3e-4` | Base阶段 `3e-4` | 数值相同，但训练阶段不同 |
| Refine 学习率 | Joint中 `3e-3` | Refine阶段 `3e-4`；Joint与Base共用 `1e-5` | Refine更新幅度不同 |
| scheduler / 预算 | StepLR，每30 epoch乘0.5；80 epochs | cosine；Base 60K steps，Refine/Joint各200 epochs | 衰减曲线和训练量不同 |
| Refine 初始化 | 原训练链 checkpoint | 由当前 E1 Base 冻结训练得到 | 起点不同 |
| best checkpoint 选择 | 按 test MSE 选 | 按 train 内 validation 选 | 选择协议不同；旧结果有 test 选择泄漏风险 |

所以原 Exp7C final `47.1141 dB` 与 E1 Joint final `46.5200 dB` 虽然在同一 test-200、同一512/256几何上评估，但训练过程不匹配，不能解释为仅由 LRNet 架构造成的差值。原 Exp7C coarse 很低、Refine final 很高，是历史联合优化路径的表现；不能据此断定“高学习率把 coarse 训坏、Refine 只恢复到原本应该达到的值”。

输入几何是训练条件的一部分，不是与训练参数无关的细节。512 全图和 128 patch 每次前向看到的空间上下文、参与计算的像素数量、边界比例都不同；对含 BatchNorm 的 LRNet，每设备 batch 也会改变归一化统计。训练 crop、每个 rank 的 batch、数据划分和每 epoch 更新数都会影响优化轨迹。E2 让 E0/E1 使用相同 128/64 patch 来控制网络间对比，回答“旧优化配方下两个网络谁更适合 patch 训练”；它不能回答“只把原 Exp7C 的网络换成 VIVNet、其余连 512 全图输入都不变会怎样”。后一个问题需要另做全图 E1 配对训练。

## E0/E1：未对齐的历史结果

下面保留旧 E0 `seed42_rerun` 与 E1 的历史指标，供追溯模型行为。E0 虽达到60K步，但实际双卡暴露量未与 E1 四卡对齐；这些数字不能作为严格结构消融。新的 E0 双卡匹配运行使用 `seed42_exposure3840000` 独立目录。

| 项目 | E0 | E1 | 状态 |
|---|---|---|---|
| 数据 | 20260818 配对2× LR TIFF / HR input / WLS target | 相同 | 对齐 |
| train/validation split | train1803、validation200，seed42 | 相同ID hash | 对齐 |
| 模型 | 旧24通道 LRNet + 原 ConvGuided | VIVNet U-Net + 原 ConvGuided | 仅 coarse 架构不同 |
| HR/LR 训练 crop | 128/64，随机对齐 | 128/64，随机对齐 | 对齐 |
| 分布式/精度 | 旧 E0 实际含双卡运行 | 四卡 DDP/NCCL，BF16 | 不对齐 |
| 每卡 batch / global batch | 旧运行不能按最终协议文件断言全程为64 | 16 / 64 | 不对齐 |
| workers / pin memory | 每卡8 / False | 每卡8 / False | 对齐 |
| optimizer / scheduler | AdamW，weight decay1e-4；cosine | 相同 | 对齐 |
| Base LR / 预算 | 3e-4→1e-6；60K updates | 相同 | 更新数一致，暴露量未对齐 |
| loss | MSE + 动态平衡 Sobel 向量 L1 | 相同 | 对齐 |
| 初始化 | 随机初始化 | 随机初始化 | 对齐，不从旧 checkpoint 续训 |
| Base checkpoint 选择 | train 内200张全图 validation MSE | 相同 | 对齐 |
| Refine 阶段 | 各自 Base 冻结，各自独立训练同构 Refine | 同协议 | 计划对齐 |
| Joint 阶段 | 各自 Base+Refine 解冻，当前共享 LR1e-5→1e-6 | 同协议 | 计划对齐 |
| 最终 test | 中心512 HR / 对齐256 LR，独立 test-200 | 相同 | 对齐；只在选定 checkpoint 后报告 |

### Base 结果

| 模型 | 最佳迭代 | Validation MSE | Validation MAE | Edge MAE | PSNR |
|---|---:|---:|---:|---:|---:|
| E0（旧 LRNet + 原 GF） | 60,000 | `8.4636617e-5` | `0.00617897` | `0.00350180` | **40.7244 dB** |
| E1（VIVNet U-Net + 原 GF） | 55,000 | `3.1830901e-5` | `0.00349649` | `0.00284793` | **44.9715 dB** |

旧 E1 比旧 E0 高 `4.2471 dB`，E0 的 MSE 约为 E1 的 `2.66` 倍。旧 E0 的 validation PSNR 从 5K 的 37.0472 dB 提升到 60K 的 40.7244 dB；由于训练暴露量不一致，不能把该差值归因于网络结构。

### 对“E0 效果很差”的具体判断

这里的“差”是旧运行的观察，不是严格同协议的 E1 Base 对照，也不表示 E0 没有学到有效映射。E0 在 5K、10K、20K、40K、60K 的 validation PSNR 分别为 37.0472、37.9137、38.4571、40.2952、40.7244 dB；best checkpoint 位于训练末尾，但训练暴露量并未对齐。

两个模型的 LR predictor 结构确实不同：E0 是 24 通道的单尺度四个 dilation block，E1 是宽度 32 的三尺度 U-Net；参数量分别为 26,461 和 902,426。但旧 E0 运行的暴露量未与 E1 对齐，因此这些结构差异目前只是待验证的解释，不能由旧指标单独确认。

E0 新配方按 E1 暴露预算重训后，才能判断 Base 阶段的结构收益。若继续比较下游系统，还需使用新 E0 Base 重新训练其 Refine/Joint，不能沿用旧 E0 Base 初始化的权重。

Refine 和 Joint 的比较回答的是“各自 Base 进入同一后续训练流程后的系统表现”，但由于 Refine 从各自 coarse 状态训练，它不是单独隔离 LR predictor 的模块消融。应同时报告 Base、Refine final、Joint coarse、Joint final，不能只看最终输出。

### E0 训练产物

E0 训练目录：`runs/scratch/exp7c_e0/seed42_rerun/`。其中 `best.pth` 和 `final.pth` 均为 60,000 步；`history.csv` 记录每5K全图 validation，`train_loss.csv` 记录每100步 loss。早先的 `seed42` 目录仍保留一次性第1步的暂停记录，不是本次结果来源。

旧 E0 checkpoint 的最终协议字段写有 global batch64，但不能证明整段训练均以该 batch 运行。新的匹配预算训练写入独立目录，不从旧 checkpoint 恢复。

## 旧 E0 与 E1 差异的待验证解释

1. **LR predictor 的表示容量和结构不同。** E0 只有 24 通道的平面 LRNet 和四个 dilation block；E1 是宽度32的三尺度 U-Net，包含下采样、跨尺度特征处理和 skip connection。E1 的有效上下文和特征组合能力更强，尤其适合输入 LR 与 HR guide 之间的局部映射。参数量也从 E0 的 `26,461` 增加到 E1 的 `902,426`，约34.1倍；ConvGuided 部分相同，差异几乎全部来自 LR predictor。
2. **两者不是“同等规模、只换一个 block”。** E0 是单尺度的串行 dilation 结构，dilation 只是扩大采样间隔；它没有 U-Net 的低分辨率全局分支和多尺度 skip。匹配预算的重训完成后才能检验这项结构差异的影响。
3. **E0 的训练没有失败迹象。** E0 validation PSNR 从 `37.0472 dB`（5K）稳定上升到 `40.7244 dB`（60K），best 在最终点；训练 loss 也降到约 `1e-4` 量级。继续按同一 schedule 到末尾没有出现断崖，因此当前差距不是 resume 错误、过早停止或 best 覆盖造成的。
4. **GF 结构相同。** E0 和 E1 使用同一个原 ConvGuided，参数量都是 `1,261`；旧运行的训练暴露量未对齐，尚不能排除该因素的影响。
5. **这还不是最终系统结论。** E0 的外接 Refine/Joint 尚未训练，不能直接把 E0 Base 的 `40.7244 dB` 与旧 Exp7C 的历史 final `47.1141 dB` 比较后下结论。下一步应按相同 Refine/Joint 协议训练 E0，再报告 E0 Base、Refine final、Joint coarse 和 Joint final。

## 运行产物与命令

- E1 Base：`runs/scratch/viv_gf_e1/seed42_ddpfix/`
- E1 Refine：`runs/three_stage/viv_gf_e1/seed42/refine/`
- E1 Joint：`runs/three_stage/viv_gf_e1/seed42/joint/`
- E0 Base 旧运行（历史参考）：`runs/scratch/exp7c_e0/seed42_rerun/`
- E0 Base 匹配预算新目录：`runs/scratch/exp7c_e0/seed42_exposure3840000/`
- E0 早期暂停记录：`runs/scratch/exp7c_e0/seed42/`
- 当前 E1 与历史 Exp7C 的 test-200评估汇总：`evaluation/test_200_512/summary.json`
- E0 Base 新运行命令：见 `TRAIN_REFINE_JOINT_CN.md` 的“E0 Base”段落；避免复用已有 seed42 目录。
- Refine/Joint 启动脚本：`scripts/run_refine_joint_ddp.sh`；通过 `VARIANT`、`BASE_CHECKPOINT`、`REFINE_DIR`、`JOINT_DIR` 分别隔离 E0/E1。

当前 test-200 汇总的复现命令和旧 checkpoint 来源记录在 `evaluation/test_200_512/README_CN.md`。E0 训练完成后，用同一评估入口传入 `--variant exp7c_e0` 和 E0 的 Base/Refine/Joint checkpoint，输出到独立目录；不能覆盖 E1/旧 Exp7C 结果。

## 结论边界与下一步

1. 旧 Exp7C 的 `33.5392 → 47.1141 dB` 显示 Refine 大幅补偿旧 coarse，但旧训练与 checkpoint 选择方式不适合做架构或学习率的单因素结论。
2. 旧 Base 结果中 E1 比 E0 高 `4.2471 dB`，但 E0 训练暴露量未对齐；匹配预算的新 E0 结果出来前，不用这项差值做结构归因。
3. 按相同 Refine/Joint 协议训练 E0；E0/E1 每阶段均按各自 train 内 validation 选择 best，并完整记录 coarse 与 final。
4. 只在训练产物完整且 checkpoint 锁定后，用独立 test-200 中心512/256 crop 做一次系统报告。test 集不参与模型或超参数选择。

## 2026-11-07：E2 首次运行作废与修复

`runs/legacy_matched_patch128/e0/base/` 首次运行完成 80 epochs，但仅约 16.60 dB，随后 Refine 初始化失败。实查 checkpoint 顶层没有 `phase`（只在 `protocol.phase` 中存在），而读取器检查顶层 `phase`，导致阶段衔接失败。

同时发现训练数值域错误：E0 的公共 coarse 接口已经返回 `[0,1]`，旧 E2 适配器没有减 0.5，却和 `GT-0.5` 计算损失，验证又给预测加 0.5。此运行不能作为结构对比结果，保留原文件追溯，禁止续训进入 Refine。

早期 v2 修正了数值域和 checkpoint phase，但其学习率和 loss 协议随后被发现不匹配，故不作为有效实验。v4 修复了阶段协议，但暴露预算仍不匹配。当前 v5 统一 `coarse_centered = coarse_raw - 0.5`，checkpoint 保存顶层 phase 和 implementation_revision=5；Base 不运行未训练的 Refine；Refine 阶段固定 coarse 的 eval 状态。v5 的完整六阶段串行链尚未启动。
# 新 A1/A2：可学习多卷积核仿射估计及边缘配对（2026-11-07）

旧 A1/A2 使用固定均值 box 和较浅系数头，旧 A1/A2 已各完成双卡每卡16、global32、60K 训练；代码及结果移到 [旧 A 归档](archive/legacy_a/README_CN.md)。旧 A1 checkpoint 可严格加载到 `viv_gf_a1_fixed_boxes`，但其 global32 与 E1 global64 不匹配。本轮串行脚本将它作为第五个变体，在新目录按 global64 从头重训；旧权重只保留为非严格匹配参考。固定单窗口 box3/box9 也保留。设计、完整训练协议和串行命令见 [LEARNED_KERNEL_EXPERIMENTS_CN.md](LEARNED_KERNEL_EXPERIMENTS_CN.md)。

| 实验 | 估计方式 | 与 E1 保持一致的部分 | 状态 |
|---|---|---|---|
| `viv_gf_a1_stats` | 可学习3×3/9×9统计卷积，各接原完整系数网络；由均值计算b；门控融合a、b | VIVNet LRNet、原3×3分支、系数网络层数/AdaptiveNorm/ReLU、HR上采样与输出 | 已实现，未训练 |
| `viv_gf_a2_stats_edge` | 统计版 A1 门控增加原图及预测 y_lr 的梯度/Otsu 边缘图；候选分支不变 | 与统计版 A1 配对，仅门控输入增加边缘信息 | 已实现，未训练 |
| `viv_gf_a1_direct` | 可学习3×3/9×9分支直接预测a、b，门控融合 | VIVNet LRNet、系数网络宽度及归一化/激活、HR上采样与输出；不保留统计量约束 | 已实现，未训练 |
| `viv_gf_a2_direct_edge` | 直接预测版 A1 门控增加相同的四张边缘图；直接预测分支不变 | 与直接预测版 A1 配对，仅门控输入增加边缘信息 | 已实现，未训练 |
| `viv_gf_a1_fixed_boxes` | 固定均值 3×3/9×9 双窗口及当前系数头 | 与 A1 learned-kernel 方案比较；当前脚本按 global64 从头训练 | 旧 global32 已训练；匹配训练待完成 |

五个 A 变体使用双卡 DDP/NCCL、每卡 micro-batch16、梯度累积2、有效 global batch64、每rank 8 workers、60K optimizer steps、HR128/LR64、BF16、AdamW及相同数据划分和 `base` loss，从头训练全部参数。固定双窗口变体不加载旧 global32 权重。样本呈现量为 `64 × 60,000 = 3,840,000` 次，与 E1 相同。E1 四卡每卡16时，每 rank 每轮 28 个完整 micro-batch；A 系列双卡每卡16时每 rank 每轮 56 个 micro-batch，梯度累积后也是28个 optimizer steps，且每轮全局处理的训练 crop 数相同。不同 world size 的 sampler 切分和随机 crop 顺序不逐样本完全一致；此外动态平衡 loss 在每个 micro-batch 上计算权重，因此累积梯度不保证与一次 global64 前向逐元素相同。可以据此称为训练样本呈现预算和优化步数对齐，不应称每一步的梯度完全相同。启动脚本在恢复或跳过已有运行前核对 E1 协议。暂无性能结果。

## A1 基线上的独立补充实验：自由 b、branch 输入和 gate（方案，2026-11-08）

这一组实验以已经存在的 `viv_gf_a1_stats` 为唯一基线。基线目录、checkpoint 和历史指标不覆盖、不改名；新实验全部使用基线名加后缀，作为 A 系列补充。旧的 `viv_gf_a2_stats_edge` 仍表示“edge 只进入原 gate”的历史定义，不能与下面的 `edgebranch` 变体混用。

### 基线的固定定义

`viv_gf_a1_stats` 保持当前实现：3×3 和 9×9 两个可学习统计 branch 分别估计 `a`，并使用各自局部统计量计算

```text
b_k = mean_y_k - a_k * mean_x_k
```

两个 affine pair 使用同一组逐像素 gate 权重融合。基线的输入、branch、`b` 约束、gate 和训练协议均不改变。这里的 `b` 具有引导滤波中的局部均值校正意义；后续 `freeb` 实验专门检验去掉这个约束的效果。

### 第一轮只改变一个因素

| 补充实验名 | 相对 `viv_gf_a1_stats` 的唯一结构/训练变化 | 保持不变的部分 | 该实验回答的问题 |
|---|---|---|---|
| `viv_gf_a1_stats_freeb` | 每个 3×3/9×9 statistics branch 的系数头由 1 个输出通道改为 2 个，直接输出 `[a_k,b_k]`；不再计算 `b_k=mean_y_k-a_k*mean_x_k`。`b` 输出通道零初始化只用于稳定起点，不恢复该均值约束 | statistics 输入、`a/b` branch 结构、edge、gate、重建方式 | 抛弃引导滤波均值约束后，独立 `b` 是否改善 affine 重建 |
| `viv_gf_a1_stats_edgebranch` | guide 与预测 `y_lr` 的梯度/Otsu 特征进入每个 branch 的系数估计器，帮助 branch 估计 `a`；`b` 仍按基线均值关系计算 | `b` 约束、原 gate、3×3/9×9窗口和重建方式 | edge 是否直接改善 branch 的局部 affine 参数估计 |
| `viv_gf_a1_stats_routesup` | branch 保持基线；训练时用两个候选 affine 输出分别与 GT 的误差生成 soft routing target，增加 gate routing loss；推理时不输入 GT | `a/b` 估计、统计输入、edge、最终融合形式和主重建 loss | gate 是否能学会根据候选 affine mapping 对 GT 的贴近程度分配权重 |

这三个变体是从同一个 A1 基线分别分叉的独立实验，不是逐项叠加的链。第一轮禁止同时改变 `b`、branch 输入和 gate，避免无法归因。新目录只在名称上添加上述后缀，已有 A 系列产物继续保留原位置。

### `routesup` 的训练与推理边界

对每个 branch 先形成候选 HR 输出

```text
z_k = up(a_k) * x_hr + up(b_k)
```

训练时分别计算 `z_3`、`z_9` 与 GT 的局部误差，经过温度 `0.05` 的 softmax 得到 routing target；gate 的输入仍只能使用推理时可获得的 `x_lr`、`y_lr`、统计量、`a`、`b`、候选映射及可用 edge 特征。GT 只用于生成辅助监督，不进入 gate 的 forward 路径。gate 输出的权重继续同时融合 `a` 和 `b`，不改成分别选择斜率和偏置。候选输出误差是主要 routing 依据；不把两个 branch 之间的 disagreement 作为核心判据。routing 辅助交叉熵的权重固定为 `1e-4`，主 `base` 重建损失和训练超参数保持不变。

`a` 和 `b` 的数值本身不应被强制拟合一组唯一的 GT 系数：在局部方差很小时，不同的参数组合可能产生近似相同的输出。因此如果后续加入 affine 参数辅助项，只能作为弱约束，并根据局部方差降低不可靠的 `a` 监督权重；第一轮的主要判断仍使用候选图到 GT 的误差和最终重建结果。

### 训练协议和后续叠加规则

三个补充实验都必须沿用 E1 新配方：相同 train/validation 划分、HR128/LR64 随机 crop、双卡 DDP/NCCL 的 A 系列设备配置、每卡 micro-batch16、梯度累积2、有效 global batch64、每 rank 8 workers、60K optimizer steps、BF16、AdamW、相同 cosine 学习率和 `base` loss。不能因为改变 `b`、branch 输入或 gate 而改变暴露量、优化器或数据协议。

第一轮结果出来后，只有在某个单因素变体确认有效时才继续训练组合版本，例如 `viv_gf_a1_stats_edgebranch_freeb` 或 `viv_gf_a1_stats_edgebranch_routesup`；三项都有效时再考虑三者组合。若三个单项都无效，应先检查 3×3-only、9×9-only、当前融合和 oracle routing 的上限，不直接继续叠加结构。
