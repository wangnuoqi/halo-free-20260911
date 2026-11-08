# Exp C: LR predictor 多尺度路由对照

## 要回答的问题

现有 DeepGuidedLR4 的四个 VIV-inspired block 在半分辨率 LR 图上顺序运行。只把这四个 block 改为编码器/解码器的多尺度路由，是否能改善 guided-filter coarse 输出？这一步不训练三阶段 Refine，也不做 joint training。

## 配对结构

| 项目 | `deep_guided_lr4` 控制组 | `guided_pyramid_lr4` 候选组 |
|---|---|---|
| 输入/输出 | 2x LR + HR guide -> HR coarse | 相同 |
| stem、四个 block、tail | 24 通道，block dilation 为 2/4/8/16 | 完全相同的模块和参数量 |
| block 路由 | 全部在 LR 尺度顺序运行 | 第 1 个在 LR，第 2 个在 LR/2，第 3 个在 LR/4，第 4 个在 LR；解码时双线性上采样并与 skip 取平均 |
| guided filter | 相同的 3x3 box statistics、系数头与仿射重建 | 相同 |

候选的降采样采用 2x average pooling，解码只用插值和 skip 平均，不额外增加可训练参数。两者的 checkpoint state-dict 键一致，但不同路由下的权重功能不等价；本轮从头训练，不把旧权重加载后称为续训。

## 筛选协议

沿用 Exp C 的 20260818 train 目录、seed=42 的 200 图验证 holdout、128x128 HR / 64x64 LR 随机 paired crop、global batch 64、BF16、AdamW、Charbonnier + 0.05 梯度 L1、60K cosine 调度。两组都在 5K optimizer updates 停止并使用同一 200 图全图验证指标。独立 test-200 不参与筛选或 checkpoint 选择。候选输出单独写在 `runs/guided_pyramid_lr4_screen/guided_pyramid_lr4/`。

控制组已有的 5K 验证为 36.693 dB，但该 run 在 25K 后出现 NaN，不能使用 60K 权重。比较时需要核对控制组 5K checkpoint 与候选的协议、有效权重和完整验证记录。若候选没有稳定优于控制组，则不进入 Refine/joint 阶段。若有 coarse 收益，再比较 coarse-only、冻结 coarse 训练 Refine、最后低学习率 joint 的全图质量与推理时间。

## joint 是否让前两阶段白训

不会。前期 coarse 训练提供初始化和独立可测的 coarse checkpoint；冻结 coarse 训练 Refine 时它保持不变。joint 解冻后 coarse 可能变成更便于 Refine 使用的中间表示，因此必须同时监控 coarse 与 final，并保留 joint 前的 checkpoint。只有在独立全图验证表明 joint final 有收益、coarse 退化可接受时才采用 joint 权重；不能把 joint 后的 coarse 当作原 coarse 模型的成绩。
