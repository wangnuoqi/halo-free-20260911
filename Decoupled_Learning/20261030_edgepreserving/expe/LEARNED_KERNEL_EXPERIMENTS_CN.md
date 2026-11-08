# 可学习多卷积核系数估计：A 系列新对照

用户确认的目标：在同一 LR 网格上使用可学习的 3×3、9×9 卷积估计两组仿射系数，用逐像素门控共同融合 a、b。后续 bilinear 上采样、align_corners=True、中心化和输出 clamp 均沿用 E1。

- A1 `viv_gf_a1_stats`：保留 E1 完整 3×3 ConvGuidedFilter 分支；9×9 分支仅扩大统计卷积，保留相同的三层 1×1 系数网络、AdaptiveNorm 和 ReLU。两个分支独立学习。各自由均值关系计算 b，再用同一组 softmax 权重融合 a、b。
- A2 `viv_gf_a2_stats_edge`：在统计版 A1 的门控输入追加四张 LR 边缘图（HR 原图和预测 y_lr 的梯度幅值及 Otsu）；统计卷积、系数分支及重建均不变。
- A1-direct `viv_gf_a1_direct`：两个分支分别以可学习 3×3、9×9 卷积读取 x_lr、y_lr，后接 AdaptiveNorm、ReLU、32→32 的 1×1 层和双通道输出层，直接预测 a、b。门控读取输入和候选系数。该版本改变统计约束，不能称为只增加窗口的严格消融。
- A2-direct `viv_gf_a2_direct_edge`：在直接预测版 A1 的门控输入追加相同的四张边缘图。边缘图只影响窗口选择，不进入两个候选分支。
- 四个新变体均从头训练，LRNet 全解冻，不加载 E1 权重，不附加外部 Refine/Joint。门控末层零初始化，初始等权，不强制固定窗口。
- `viv_gf_a1_fixed_boxes` 也纳入本轮串行训练，作为第五个变体。使用固定 3×3/9×9 均值统计候选和可学习分配/斜率头，VIVNet LRNet 从随机初始化开始并全程解冻；不加载旧 A1 checkpoint。
- 与已完成 E1 Base 对齐训练预算：推荐双卡 DDP/NCCL（默认物理卡2、3），每卡 micro-batch16、梯度累积2，有效 global batch64；60K optimizer steps，因此样本呈现量与 E1 同为3.84M。AdamW、LR3e-4、minLR1e-6、weight_decay1e-4、BF16、seed42、每rank 8 workers、pin_memory=False。数据、HR128/LR64、验证划分和动态平衡 MSE/Sobel loss 相同。E1 实际是四卡、每卡16、累积1；新命令的每卡 micro-batch 与有效 global batch相同，但 world size 不同，采样分片/批次组成不保证逐步完全相同。
- 原 A1/A2 固定均值版本及结果移到 [`archive/legacy_a`](archive/legacy_a/README_CN.md)：旧 A1 checkpoint 是双卡每卡16、global32、60K，作为旧协议参考。本轮串行脚本会在独立目录按有效 batch64从头重训 `viv_gf_a1_fixed_boxes`，新结果与 E1 的样本呈现量匹配。固定单窗口 `viv_gf_a1_box3`、`viv_gf_a1_box9` 仍保留。

新顺序为 `A1-stats -> A2-stats-edge -> A1-direct -> A2-direct-edge -> A1-fixed-boxes`。E1 对 A1-stats 检验可学习双窗口模块整体；两条 A1 对 A2 配对各自检验边缘信息；两条 A1 之间检验统计约束与直接预测。A1-direct 对 E1 同时改变估计方式及窗口模块，不作单因素归因。固定双窗口作为传统 box 统计基线放在最后。

实现步骤：

1. 先增加 CPU 回归测试：门控仅选择3×3时与 E1 等价；两候选和门控梯度存在且有限；输出尺寸及模型注册正确。
2. 在 `expe/learned_affine.py` 实现两个系数估计器，在 `expe/model.py` 注册并接入既有重建路径。
3. 增加双卡和四卡串行脚本，依次运行四个新统计/直接预测 A 变体及固定双窗口对照，自动从各自 latest 恢复、完成后跳过，并与 E1 的实际训练协议核对。固定双窗口旧 A1 仅作为 global32 参考；匹配运行使用全新目录从头训练。
4. 执行 CPU 测试、相关既有模型测试和 shell 语法检查。不开启 CUDA 自检或实际训练。

结果仍需完整图验证和独立512 test-200；训练完成前没有性能结论。

## 启动命令

```bash
cd /mnt/d/ResearchCode/halo-free/Decoupled_Learning/20261030_edgepreserving
CUDA_VISIBLE_DEVICES=2,3 bash expe/scripts/run_a_learned_kernels_ddp_2gpu.sh
```

依次训练 A1-stats → A2-stats-edge → A1-direct → A2-direct-edge → A1-fixed-boxes，各60K。双卡每卡micro-batch16、累积2，有效batch64，8 workers/每个rank。默认读取现有 20260818 数据和 `expc/runs/0818_memmap_4gpu/train_0818_float32.npy` 缓存。若希望四卡每卡16、累积1，也可运行 `run_a_learned_kernels_ddp_4gpu.sh`，两种方式有效batch与优化步数相同。

各目录含 final 时跳过，含 latest 时传入 `--resume`，否则从头训练。目录分别为：

- `expe/runs/a1_a2_e1_matched_2gpu_accum2/viv_gf_a1_stats`
- `expe/runs/a1_a2_e1_matched_2gpu_accum2/viv_gf_a2_stats_edge`
- `expe/runs/a1_a2_e1_matched_2gpu_accum2/viv_gf_a1_direct`
- `expe/runs/a1_a2_e1_matched_2gpu_accum2/viv_gf_a2_direct_edge`
- `expe/runs/a1_a2_e1_matched_2gpu_accum2/viv_gf_a1_fixed_boxes`

训练入口复用现有 tqdm 进度条、`train_loss.csv`、`history.csv`、`protocol.json` 和 best/latest/final checkpoint。LRNet、两个候选分支和门控全部参与训练。为 VIVNet 上游未使用参数保留原入口的 DDP unused 参数检测。

脚本读取 `seed42_ddpfix/protocol.json`，检查 E1 的并行、batch、数据路径、优化器、loss、训练预算和初始化设置；已有运行还会核对训练/验证划分哈希，拒绝把旧双卡 checkpoint 作为本组结果续训或跳过。`num_workers=8` 和 `save_every=100` 未记录在 E1 protocol 中，按 E1 正式训练命令对齐。

统计量版本沿用 E1 可学习卷积的有符号归一化分母，没有额外非负约束或 epsilon；这保持了原计算，也保留了分母接近零时可能不稳定的性质。应观察loss及候选系数，不提前断言大窗口一定改善。

## 已完成验证

- CPU 回归：强制仅选3×3分支后，统计版 A1 与原 E1 guided filter 输出逐元素一致。
- CPU 回归：两种 A2 在复制对应 A1 权重并输入零边缘图时，候选系数和门控输出一致；非零边缘图可改变门控权重。
- CPU 反向传播：四个新变体的完整模型输出几何和 LRNet 反向传播正常；两种估计方式的分支、门控梯度存在且有限。固定 box 对照复用原 MultiBoxAffine 实现。
- 尚未启动正式训练，也未验证四卡 NCCL 的实际吞吐和显存。
