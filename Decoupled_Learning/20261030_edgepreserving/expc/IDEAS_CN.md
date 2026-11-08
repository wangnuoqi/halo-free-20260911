# Exp C：面向 WLS 保边滤波模拟的可证伪方向

## 问题定位

任务不是一般图像去噪，而是从单通道 log-intensity 输入预测固定协议的 WLS base，训练数据来自 `S2R-HDR/data/20260818`，标签为 `lambda=0.5, alpha=1.4`。现有 NAFNet 与 VIVNet 都能高质量近似标签；原三阶段模型即使参数少，也可能被高分辨率串行算子和多个小 kernel launch 拖慢。因此 Exp C 不再延续“三阶段里继续加 block”的路线，而是直接研究不同的归纳偏置能否改善质量、速度和可解释性。

## 实验方案

| ID | 实验 / 可证伪假设 | 结构和可分析量 | 预期风险 |
|---|---|---|---|
| C1 `naf_small` | 窄化的两尺度 NAF 式 encoder-decoder 可作为干净容量控制组；若仍明显低于 NAFNet，单纯缩小通用复原网络不足以满足目标。 | 约 151K 参数；记录 encoder/decoder 激活、参数量、全图 latency。 | 容量太小，可能速度好但 PSNR 落后。 |
| C2 `similarity_pyramid` | 跨尺度 skip 是否相似，应该决定浅层细节传多少；显式相似度门比无条件 skip concat 更适合保边目标。 | 约 117K 参数；可视化 cosine similarity、edge guide 和逐像素 gate。 | 学到的 gate 可能退化为常量；必须与关闭 gate 的消融比较。 |
| C2b `wls_naf` | 多尺度相似度融合、边缘修正和单次 residual 输出，能否在保留通用复原表示能力时拟合 WLS。 | 三尺度 NAF encoder-decoder；两层 similarity fusion；full-resolution edge correction；记录各分支误差和 gate。 | 结构比轻量候选大；需用实测 512 延迟判断是否有部署价值。 |
| C3 `lowres_edge` | WLS 输出的主体是平滑 base；先在 1/4 分辨率预测 base，再只在高频区域做小修正，能减少昂贵的全分辨率特征计算。 | 约 39K 参数；检查 low-res base、高频残差和 edge gate。 | 低分辨率会丢失窄边界，detail branch 可能补不回来。 |
| C4 `unrolled_diffusion` | 学习边缘导通率并展开固定步数的 WLS 型局部更新，可以用很少参数获得可解释的滤波近似。 | 约 2.9K 参数；保存水平/垂直 conductance、每轮中间结果和有效 lambda。 | 固定局部迭代的感受野有限；质量可能受限，迭代也可能形成串行延迟。 |
| C5 `deep_guided_lr4` vs `deep_guided_lr8` | 在固定通道数、block 类型、分辨率和 guided-filter 后处理下，把 LR predictor 深度从 4 个 block 增至 8 个，能否改善 WLS 拟合？ | 两者均 24 通道，dilation 周期 `[2,4,8,16]`，使用相同 VIV-inspired block、ConvGuided 系数头和原始 2x LR / HR guide 配对；记录参数量、全图延迟、PSNR、edge error。 | 深层大 dilation 可能重复/稀释局部证据并增加延迟；若只比较不同协议 checkpoint，不能作因果结论。 |

五种模型均直接预测 WLS base，不保留 ConvGuided + 三阶段 Refine。它们训练时共用同一数据划分、patch、batch、optimizer、训练步数和 loss：Charbonnier 像素项加 `0.05 *` 一阶有限差分梯度 L1。这样新实验间的差异主要来自结构。训练集内固定 200 张验证图用于选 best；独立 0818 test-200 只在 best 确定后评估。

目前五个候选共用 0818 train split。第一轮四个候选已完成 5K 筛选；`naf_small`、`similarity_pyramid` 和 `lowres_edge` 正继续训练，`unrolled_diffusion` 因 5K 验证明显落后且进程停滞而暂停，保留其 checkpoint。`wls_naf` 正在独立 tmux 队列中等待四张 GPU 空闲，再启动四卡 DataParallel 正式训练。所有数字应以各自 `history.csv` 与 checkpoint protocol 为准。

## 比较协议

现有强基线训练协议并不完全一致：NAFNet checkpoint 使用 256 crop、3 卡每卡 batch 32、PSNR loss，记录到 60K iteration；VIVNet checkpoint 使用 128 crop、4 卡每卡 batch 48、L1 + 0.1 FFT loss，best 在 20K iteration。二者使用相同 0818 数据及相同 deterministic train/val ID split，但新 Exp C 的通用 loss 不同。Exp C 采用 128 crop、4 卡每卡 batch 16（global batch 64）、60K iteration，patch 数量与 VIVNet best checkpoint 的 20K × global batch 192 相同；比 NAFNet 记录的 60K × global batch 96 少约三分之一。由于 crop 面积与 loss 不同，新 Exp C 与旧 NAFNet/VIVNet 比较仍是“当前最佳模型对新方法”的实际比较，不应夸大成完全相同优化协议下的架构因果结论。

主要质量指标：test-200 的 `PSNR from mean MSE`、逐图 PSNR 平均、MAE、边缘梯度误差。效率指标：batch=1 全图 FP32 latency（预热后同步计时）、参数量，并记录 512 / 原生尺寸。C2 另报告 gate 与 edge 的相关性；C3 记录 base/detail 分支误差；C4 记录每次迭代残差、conductance 图。若候选方法没有进入 NAFNet/VIVNet 的质量—延迟 Pareto 前沿，则不继续堆模块。

## 决策门

1. WinTrain 的 DataParallel formal/smoke 路径使用四张卡；为提高结构筛选吞吐，首轮可并行运行四个 single-GPU 候选，每个候选一张卡，且各自保持 global batch 64。
2. 对有潜力的候选继续完整 60K-step 四卡 DataParallel 训练，固定 0818 数据和 200 张 validation holdout。
3. 每个 best checkpoint 用未参与选择的 `test-200` 做整图评估，报告速度、PSNR、edge error 和参数量。
4. 首轮只晋级 1 个候选做针对性消融：C2 gate ablation、C3 1/2 vs 1/4 base scale，或 C4 迭代数 2/4/6。不能用 test-200 调超参。
5. C5 的 4/8 block 对照单独在相同 Exp C 训练协议下先跑 5K validation screening；若 8-block 没有验证收益或延迟明显恶化，就停止加深。仅验证集筛选后才考虑 60K 正式预算，不用 test-200 选深度。

## 文献依据

- Chen et al., “Simple Baselines for Image Restoration,” ECCV 2022, arXiv:2204.04676. NAFNet 说明 image restoration 的强基准可由简洁、硬件友好的 block 取得；因此 C1 是轻量容量控制，不能把“用了 NAF block”本身当贡献。
- Zamir et al., “Restormer: Efficient Transformer for High-Resolution Image Restoration,” CVPR 2022, arXiv:2111.09881. 其多尺度 encoder-decoder 与高效跨通道交互支持多尺度建模假设；C2 把问题收窄到“skip 信息是否与目标尺度相似”。
- Zamir et al., “Multi-Stage Progressive Image Restoration,” CVPR 2021, arXiv:2102.02808. 渐进式多阶段是常用恢复范式，但当前三阶段系统的延迟结果提示，阶段数本身不是优势；C3/C4 检验能否用单次多尺度预测或固定优化迭代替代串行 learned stages。
- Cui et al., “Visual-in-Visual: A Unified and Efficient Baseline for Image Restoration,” IEEE TPAMI 2026, DOI:10.1109/TPAMI.2026.3669720. VIVNet 使用多感受野、similarity-aware weighting 与高阶交互；本项目此前只将受启发的 micro-block 放入 LR predictor，没有替换完整宏观数据流。C2 把 similarity 用作可观察的跨尺度信息选择机制。

这些文献提供设计线索，不构成 Exp C 的新颖性证明。是否有贡献由上述 Pareto 结果和结构消融决定。

## 训练可靠性待办：验证步 checkpoint 的 best 状态滞后

### 已确认的故障与影响

`src/train_expc.py` 在同一个验证步先保存 `latest.pth`，再计算验证指标并更新内存中的 `best_mse` 和 `best.pth`。因此该步的 `latest.pth` 保存的是验证前的 `best_mse`。若恰好在该步结束后中断并从 `latest.pth` 恢复，下一次验证会拿过期阈值判断 `best=True`，可能用更差的权重覆盖 `best.pth`。这是 checkpoint 选择错误；`history.csv` 中已记录的 MSE、MAE、PSNR 数值不受这个标志影响。

2026-11-03 的可复核实例：`naf_small` 在 5000 步验证 MSE 为 `0.0001152422`（PSNR `39.3839 dB`），10000 步 MSE 为 `0.0020388485`，15000 步为 `0.0005041168`；后两次仍被标为 `best=True`，目前 `best.pth` 指向 15000 步。5000 步的 `final.pth` 尚在，包含当时的有限模型权重及正确的 `best_mse`，须保留，不能用当前 `best.pth` 代替历史最佳。

### 实施方案（恢复训练前）

1. 先停止会写入目标 run 目录的训练进程，备份 `latest.pth`、`best.pth`、`final.pth` 和 `history.csv`，记录每个文件的 iteration 与 MSE；不要在运行中原地改 checkpoint。
2. 在 `src/train_expc.py` 的每个验证步，先计算验证指标、更新 `best_mse`、按真实改善情况保存 `best.pth` 并写入 `history.csv`，最后才原子写入 `latest.pth`。非验证步保持现有 `save-every` 频率。这样 `latest.pth` 的模型、iteration、优化器、调度器和 `best_mse` 始终对应同一个已完成步骤。
3. 恢复旧 run 时，校验 `latest.pth` 中的 `best_mse` 与历史记录：若存在更小的有效验证 MSE，先找到**同一步**的模型权重，再修正最佳权重和恢复阈值；不能只改 `best_mse` 而没有对应权重。`naf_small` 应从保留的 5000 步 `final.pth` 提取模型权重，重建真正的 `best.pth`；训练状态仍从有效的 `latest.pth` 恢复，并使用历史最小 MSE 作为比较阈值。修复时保留旧文件副本和来源记录。
4. 增加最小回归测试：在验证步保存后立即读取 `latest.pth`，断言 `best_mse` 等于迄今验证 MSE 最小值；模拟在该步中断并恢复，给出更差的下一次验证结果，断言 `best.pth` 的 iteration、权重和最佳 MSE 均不变；再测试更好结果能够更新两份 checkpoint。验证 `history.csv` 的 `best` 标志与实际最小值一致。

完成这些检查前，`best.pth` 仅代表文件当前内容，不代表历史最优；选模型以 `history.csv` 的真实指标和可对应的权重文件为准。此问题独立于 `wls_naf` 已出现的 NaN 数值发散，不能通过修正保存顺序消除 NaN。
