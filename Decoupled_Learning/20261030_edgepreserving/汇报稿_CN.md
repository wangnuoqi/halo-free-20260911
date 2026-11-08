# Edge-Preserving WLS 实验汇报稿

## 一句话结论

我们沿着“先确认现有模型的质量与速度瓶颈，再定位误差来自哪里，随后逐步验证结构改造”的路径推进了 edge-preserving WLS 任务。当前证据显示：Exp7C 的推理并不慢，主要短板是与 NAFNet/VIVNet 的输出质量差距；只在 LR predictor 内替换 VIV-inspired block 并未弥补差距。误差图进一步显示 Exp7C 在平滑区域和边缘区域都存在更大的误差，边缘误差尤为明显。基于此，Exp B 对 block 替换做了更受控的检验，Exp C 则转向直接拟合 WLS 的不同宏观结构。后两条路线仍需用独立测试集和可靠 checkpoint 规则完成验证，现阶段不能宣称已超过强基线。

## 可直接口头汇报的版本

我们的目标是用轻量模型近似固定参数下的 WLS 保边滤波，同时兼顾输出质量和推理效率。数据来自 20260818 WLS 数据集，包含 2003 组训练样本和 200 张独立测试图，LR 与 HR 的空间倍率为 2，任务参数为 `lambda=0.5, alpha=1.4`。

首先，我们用同一批 200 张测试图核对了已有质量基线：NAFNet 为 49.434 dB，VIVNet 为 48.992 dB，旧 Exp7C 为 47.114 dB。我们也统一测试了 512×512 单图的纯 GPU 前向：Exp7C 为 13.613 ms，VIVNet 为 54.583 ms，NAFNet 为 65.633 ms。也就是说，之前观察到 Exp7C 比它们慢的现象没有在统一单卡前向测试中复现；Exp7C 参数量更小，推理也更快，当前主要问题是质量差距，而不是模型本身的前向速度。

接着，我们想判断把 VIVNet 的优质 block 放进原有 LR predictor 是否能提升结果。第一轮 Exp A 的实验暴露出训练输入处理错误：HR 在进入模型前已经中心化，却又被重复减去 0.5；另外 stage 的 Sobel 特征归一化也和旧 Exp7C 不一致。因此，A0、A1 和旧 C1 的极低分数不能用于判断结构优劣。修正实现后，旧 Exp7C 可以在 200 张测试图上复现到 47.114 dB，确认评估和权重映射路径正确。随后修正输入的 VIV-LR 迁移实验达到 39.255 dB；延长训练后达到 40.164 dB，但仍明显低于 Exp7C。这个结果说明仅替换 LR predictor 内部 block 并不足以恢复甚至超过基线；它还不能单独证明 VIV block 无效，因为模型数据流、训练初始化和优化过程仍会影响结果。

为了找到差距出现在哪里，我们在 8 张代表图上生成了输出对比图和 40 张误差热力图。按这 8 张图的逐图误差均值，Exp7C 的灰度 MAE 为 0.002811、边缘 MAE 为 0.000775；NAFNet 分别为 0.001609 和 0.000373；VIVNet 分别为 0.001563 和 0.000323。可视化与统计一致：Exp7C 在边缘附近的误差尤其明显，但平滑区域误差也高于两个基线。因此，改进方向不能只盯着边缘锐化，也要改善整体 WLS 映射精度。

基于这个诊断，Exp B 固定 guided-filter 和三阶段 refine 主体，只替换 LR predictor 中四个 dilation block 的内部结构，并从头按 base、refine、joint 顺序训练。这比 Exp A 更接近“只测试 block 替换”的问题。训练已完成，但目前报告的是固定训练验证集中心 crop 的结果：joint best 为 41.555 dB，不能与 test-200 的 47.114/48.992/49.434 dB 当作同一协议下的最终排名。我们还在同一验证集上做了不重新训练的公式对照：保持系数不变的 affine anchor 得到 41.546 dB；边缘感知系数调整在不同强度下约为 41.547–41.554 dB，均未超过 41.555 dB 基线。初步看，这些简单公式修正没有带来有意义的收益。

之后，Exp C 把问题从“guided-filter 里继续换 block”扩展到“是否应该直接学习 WLS 映射”。候选包括小型 NAF、相似度金字塔、低分辨率 base 加边缘修正、可展开扩散，以及加深 guided-filter LR predictor。共享验证集上的早期曲线显示部分候选在 5K–20K 步达到约 39–40 dB，但随后有明显回落；`naf_small` 的 checkpoint 保存顺序还曾导致较差权重被误标为 best。`deep_guided_lr4` 在 23.3K 步出现 NaN，之后仍继续写出 NaN loss/validation；截至 2026-11-04，tmux 进程仍运行到约 26.4K 步，但这不是有效训练结果。`wls_naf` 的一次 WinTrain 运行则因 LLVM out-of-memory 失败。Exp C 目前提供的是结构筛选线索和训练可靠性问题，不是独立测试集上的最终结论。

所以目前最稳妥的阶段性结论是：Exp7C 是紧凑而快速的基线，但质量低于 NAFNet/VIVNet；把 VIV-inspired 微结构插入原 LR predictor 没有解决这个质量差距。误差分析支持我们继续研究更合适的宏观信息路径或直接 WLS surrogate，但必须先保证输入处理、checkpoint 选择和评估协议可靠。下一步应先修复并核验 Exp C 的 checkpoint/非有限值处理，再依据同一验证门筛选候选；候选固定后才跑独立 test-200，并与 Exp7C、NAFNet、VIVNet 作全图同口径比较。

## 核心结果表

| 阶段 | 研究问题与尝试 | 主要观察 | 当前结论 |
| --- | --- | --- | --- |
| 基线核验 | 对比 Exp7C、NAFNet、VIVNet 的测试质量及 512×512 单图前向 | test-200 PSNR：47.114 / 49.434 / 48.992 dB；单卡前向：13.613 / 65.633 / 54.583 ms | Exp7C 的质量有差距，但纯前向更快；旧多卡慢速结果不是单卡模型计算速度 |
| Exp A：VIV-LR 探索 | 换 LR predictor block、loss 与训练方案 | 发现 HR 重复中心化及 Sobel 归一化不一致；修正后旧 Exp7C 可复现。修正版迁移训练为 39.255 dB，延长训练为 40.164 dB | 错误输入训练的 A0/A1/C1 结果作废；单纯换 LR block 未补上质量差距 |
| 误差诊断 | 对 8 张代表图比较输出、灰度误差和边缘误差 | Exp7C 平均灰度 MAE 0.002811、边缘 MAE 0.000775；NAFNet 为 0.001609/0.000373；VIVNet 为 0.001563/0.000323 | Exp7C 的误差不只集中在边缘，但边缘误差差距更突出 |
| Exp B：受控 block 替换 | 固定 guided filter 和三阶段 refine，仅换四个 LR block，base→refine→joint 从头训练 | 固定验证 crop 上 joint best 为 41.555 dB；anchor 和 edge-aware 后处理约 41.546–41.554 dB | 比较更聚焦，但尚无独立 test-200 结果；公式小改动收益可忽略 |
| Exp C：宏观结构探索 | 比较直接 WLS surrogate、低分辨率 base、相似度路径和更深 LR predictor | 早期验证约 39–40 dB 后出现回落；部分 run 有 NaN 或 OOM；`naf_small` 曾受 best checkpoint 记录错误影响 | 作为探索阶段保留；当前没有可用于最终模型排名的统一 test 结果 |
| Exp D：速度核验 | 统一单卡、batch=1、FP32、512×512 测纯 GPU 前向 | Exp7C 13.613 ms，约为 VIVNet 的 0.25 倍、NAFNet 的 0.21 倍 | 当前优先研究质量与稳定性；正式发表绝对 latency 前应在 GPU 空闲时复测 |

## 结果口径与限制

- Exp7C、NAFNet、VIVNet 的 PSNR 来自同一 test-200 的历史全图评估；NAFNet 使用 BF16，VIVNet 使用 FP32 并补齐到 4 的倍数，Exp7C 使用偶数边裁剪，存在轻微边界协议差异。
- Exp A 修正版迁移分数来自 test-200 全图推理；1000 epoch 结果为 40.164 dB。它是迁移实验，不是严格从头训练的 block-only 因果对照。
- 热力图统计来自 8 张代表图，和 200 张正式测试集不是同一统计口径，不应把数值混入 test-200 排名。
- Exp B 的 41.555 dB 来自固定训练验证集中心 crop，必须等待独立 test-200 才能与强基线作最终比较。
- Exp C 候选目前的历史 best/验证数字受训练发散和 checkpoint 顺序问题影响；引用时必须指出 run、步数和对应权重，不直接用 history 的 `best=True` 作为模型有效性的证明。
- Exp D 的纯前向测量是在 Exp C 训练并发时采集，速度排序较稳定，但绝对延迟可能受 GPU 争用影响。

## 图表与证据位置

- 8 张代表图的原图、GT、Exp7C、NAFNet、VIVNet 和并排对比：`exp_a/evaluation/test_fei_modelvis/visualization/wls_l0p500_a1p400/`
- 灰度/边缘误差热力图（40 张）：`exp_a/evaluation/test_fei_modelvis/wls_l0p500_a1p400/gray_heatmaps/`
- 热力图汇总指标：`exp_a/evaluation/test_fei_modelvis/wls_l0p500_a1p400/gray_heatmaps/gray_error_metrics.csv`
- 正式 test-200 基线比较与评估口径：`exp_a/evaluation/comparison/README.md`
- Exp B 训练验证记录：`exp_b/runs/exp_b_vivblock_20261102/{base,refine,joint}/loss_history.jsonl`
- Exp B affine 公式对照：`exp_b/evaluation/guided_affine_anchor_joint_best_val/summary.json` 和 `exp_b/evaluation/guided_affine_edgeaware_joint_best_val/summary.json`
- Exp C 的设计假设、候选筛选门和 checkpoint 故障说明：`expc/IDEAS_CN.md`
- Exp D 的推理计时原始记录：`expd/results/inference_fp32.json`
