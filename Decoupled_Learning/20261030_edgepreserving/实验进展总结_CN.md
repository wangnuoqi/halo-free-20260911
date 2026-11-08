# 保边滤波实验进展总结

## 数据与任务

当前对比围绕 /mnt/d/ResearchCode/S2R-HDR/data/20260818 的 WLS 保边滤波数据。数据目录实查为 2003 组训练 LR/HR/GT 和 200 张独立测试图；HR 对 LR 的空间倍率为 2。历史全图测试记录为 NAFNet 49.434 dB、VIVNet 48.992 dB、旧 exp7C/C0 47.114 dB，详见 exp_a/evaluation/comparison/README.md。

## Exp A：训练方案和 LR block 的前期验证

Exp A 对训练流程、loss 组合和 LR predictor 进行了初步对照。此前发现：

1. **整体训练流程需要重塑。** 不能把新 LR predictor 随意接到旧 ConvGuided / Refine 权重上，再把结果当作公平对照；不同部件训练时的输入分布和初始化协议必须匹配。
2. **小 patch 能提高训练吞吐。** 128×128 HR/GT patch 与对齐的 64×64 LR patch 可用于训练；最终评价仍应使用全图推理。
3. **旧 C1 性能下降的原因尚未隔离。** 已知旧 C1 不是“只替换 LRNet 内部 block”的严格对照，且训练数据流曾有错误。修正输入后的 exp7C transfer 在 200 epoch 得到 39.255 dB；延长到 1000 epoch 得到 40.164 dB，仍低于旧 exp7C。当前证据不能区分未收敛与架构/训练组合不匹配。

因此，Exp A 的旧 C1 不能用来判断 VIV-inspired block 本身是否有效。A0/A1 和首次 C1 transfer 的有效性限制及历史结果记录在 exp_a/evaluation/comparison/README.md。

## Exp B：严格替换内部 block 并完整重训

### 研究问题

在 ConvGuided 和三阶段 refine 结构不变时，只把 LR predictor 的四个内部 dilation block 替换成 VIV-inspired block，完整从头训练后能否提高最终效果？

### 结构约束

- LR predictor 保留 1 通道输入、24 通道中间宽度、dilation 2/4/8/16 和单通道 raw 输出，不加 sigmoid。
- 只替换四个内部 24→24 block；使用 3×3 / 9×9 depthwise 分支、similarity weighting 和高阶乘法交互。
- ConvGuided 保留原局部 box statistics 和逐像素仿射重建。
- 三阶段 Refine 保留 delta / a / b / gate 递推。
- 不加载旧 exp7C、C0、NAFNet、VIVNet 或 Exp A 的权重。

### 分阶段训练

    Base：从随机初始化训练 VIV LR predictor + ConvGuided
      ↓ 加载 base_best，冻结 base
    Refine：训练原始三阶段 Refine
      ↓ 加载 base_best + refine_best
    Joint：解冻完整模型联合训练

- Base loss：MSE(coarse0, GT) + 0.1 × Sobel-gradient L1。
- Refine loss：0.1 / 0.3 / 0.6 加权 delta-space edge loss，目标统一为 GT - coarse0。
- Joint loss：同样的 delta-space cascade loss，加 0.5 × SmoothL1(coarse0, GT)。
- 数据：固定 seed 从 2003 组 train 中留出 200 组 validation，1803 组训练；test-200 不用于选 best。
- 几何：训练 HR/GT 128×128、LR 64×64 paired patch；验证使用固定中心 crop；最终测试全图并裁掉底部/右侧奇数边。
- 默认运行参数：四卡 DDP + NCCL、每卡 batch 16、global batch 64、BF16、AdamW 3e-4 → 3e-6 cosine、每 rank 4 workers、pin_memory=False。

### 当前实现和验证状态

代码位于 exp_b/：

- src/model_exp_b.py
- src/data_exp_b.py
- src/train_exp_b.py
- src/evaluate_exp_b.py
- run_sequence_ddp.sh 和三个单阶段启动脚本
- README_CN.md 与 HANDOFF_CN.md

已实查数据三元组数量及标签文件名映射；base、refine、joint 各自的 CPU 单步 smoke test 均成功，并生成 checkpoint。四卡 GPU 正式训练尚未启动，Exp B 测试指标尚未产生。

默认 checkpoint 位于：

    exp_b/runs/exp_b_vivblock_20261102/{base,refine,joint}/

运行命令及恢复方法见 exp_b/README_CN.md。

## 下一步

1. 按 README 中的命令顺序运行 base、refine、joint。
2. 确认每个阶段的 best/latest/final、loss_history.jsonl 和协议文件均正常保存。
3. 用 evaluate_exp_b.py 对 test-200 全图评估，只保存指标。
4. 将 Exp B coarse0、coarse1、coarse2、final 与历史 exp7C、NAFNet、VIVNet 结果并列，判断 block 替换本身及三阶段 refine 的增益。

Exp B 的效果目前待测，不预设会超过任何历史 baseline。

## Exp C：checkpoint 选择可靠性

2026-11-03 确认 `expc/src/train_expc.py` 在验证步先写 `latest.pth`、后更新 `best_mse`，从该断点恢复可能把较差模型误标为 best 并覆盖 `best.pth`。`naf_small` 的 5000 步历史最佳权重仍保存在 `final.pth`；当前 `best.pth` 不应直接用作历史最佳。具体证据、修复顺序、旧 checkpoint 恢复方法和回归测试标准见 `expc/IDEAS_CN.md` 的“训练可靠性待办”一节。此项为恢复训练前的实现待办，目前尚未修改训练代码或 checkpoint。
