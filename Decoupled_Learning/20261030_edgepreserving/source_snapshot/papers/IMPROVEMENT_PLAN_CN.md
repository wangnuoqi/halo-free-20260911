# Edge-Preserving 改进方案

## 1. 当前结论

现有结果不是“3-stage 完全没有价值”，而是当前比较同时暴露了两个问题：

1. **通用恢复 backbone 的容量和优化效率差距很大。** 当前已验证结果是
   NAFNet 60k checkpoint 为 `49.4338 dB`，exp7C 三阶段为 `47.1141 dB`。
   NAFNet 的结果来自 `baseline/evaluation/nafnet_wls_test/summary.json`，
   exp7C 来自 `baseline/evaluation/exp7c_test/summary.json`。
2. **没有把“网络结构收益”和“deep guided filter / 3-stage 先验收益”拆开。**
   目前不能仅凭最终 PSNR 判断是 backbone、loss、训练预算，还是边缘模块导致差距。

因此第一步不是直接把 VIVNet 或 NAFNet 整体替换进 3-stage，而是做可归因的
对照实验。

## 2. 论文中可迁移的要点

### NAFNet

本地全文是 `NAFNet_2204.04676.pdf`。论文的核心消融链是：

- PlainNet + LayerNorm + GELU + Channel Attention；
- 用 **SimpleGate** 替换 GELU；
- 用 **Simplified Channel Attention** 替换原 Channel Attention；
- 得到无显式激活的 NAFBlock，并放入单阶段 U-Net。

论文表 1 和表 2 显示，LayerNorm、门控乘法和简化通道注意力分别带来可测收益；
表 3 显示增加 block 数量有效，但超过约 36 个 block 后收益变小而延迟明显上升。
论文还强调 full-image inference 与 TLC/边界处理，不能只用 patch 测试结论替代全图测试。

对本任务的含义是：NAFNet 的优势很可能主要来自稳定的 U-Net 多尺度路径、
高容量的 NAFBlock、SimpleGate/SCA 和训练细节，而不是某个专门的边缘滤波模块。

### VIVNet

VIVNet 的本地论文是 `Visual-in-Visual_ A Unified and Efficient Baseline for Image
Restoration.pdf`，已核验为有效的 19 页 PDF；论文元数据在
`VIVNet_TPAMI_2026.metadata.json`。官方代码仓库为
`https://github.com/c-yn/VIVNet`。当前本地实现的主干可以直接核对
`baseline/vivnet_wls/VIVNet_arch.py`：

- OverlapPatchEmbed；
- 三层 encoder-decoder U-Net；
- bilinear upsample + convolution；
- 多尺度 Fuse；
- Restormer 风格的 attention block、LayerNorm 和 gated feed-forward；
- refinement blocks；
- global residual `output(features) + input`。

论文正文的消融实验给出了更具体的证据：以去掉 similarity-aware weighting
和 high-order processing、两个 3x3 编码分支的 Net(a) 为基线（31.40 dB），加入
similarity-aware weighting 提升 3.37 dB，加入 high-order processing 提升 3.01 dB；
两者联合为完整设计。高阶交互使用逐元素乘法，最终采用 interaction order 4；
order 5 的收益已经变慢。论文还报告 9x9 编码核优于 11x11，乘法优于加法（加法
版本低 0.24 dB），而用额外 basic block 匹配参数量的控制组只有 32.07 dB。
这些数字是论文消融结果，不能直接当成本任务的结果，但说明迁移时应优先保留
similarity weighting 和 high-order multiplication，而不是先盲目增加 block 数。

论文还说明，VIVNet 的主要贡献是把一个
biologically inspired 的 **micro visual module** 放进 macro U-shaped backbone：

- 多感受野 depth-wise convolution；
- similarity-aware weighting；
- 通过迭代 element-wise multiplication 建立高阶交互。

因此 VIVNet 的强项不是单纯的 U-Net 外形，而是每个 block 的局部多尺度、相似性
加权和乘法交互。它本身没有把 deep guided filter 或显式边缘约束作为核心机制；
guided filter/3-stage 的收益仍必须在我们的单通道 edge-preserving 数据上独立验证。

## 3. 基于两篇论文的最终方案：按 stage 职责迁移有效机制

### 3.1 论文证据与迁移边界

NAFNet Table 2 从它自己的 LayerNorm + GELU + Channel Attention baseline 开始：

| 改动 | SIDD PSNR | GoPro PSNR |
|---|---:|---:|
| baseline | 39.85 | 32.35 |
| GELU -> SimpleGate | 39.93（+0.08） | 32.76（+0.41）|
| CA -> SCA | 39.95（+0.10） | 32.54（+0.19）|
| 两者都改 | 39.96（+0.11） | 32.85（+0.50）|

这说明特征乘法门控和轻量通道重标定在 NAFNet 的 block 里有效，但不能推出
“把我们当前 GroupNorm/ReLU 直接替换为 LayerNorm/SimpleGate 必然提升”。

VIVNet 在 dehazing tiny 模型上报告：Net(a) 为 31.40 dB；加入 SW 增加 3.37 dB，
加入 HP 增加 3.01 dB；SW+HP 是完整 BIM。论文选择 3x3/9x9 分支和 interaction
order 4；order 5 收益变慢，HP 乘法比加法高 0.24 dB。这些数值不能作为本任务的
增益预期，但说明多感受野之间的相似性选择和乘法交互值得验证。

关键不是把两种 block 全部复制，而是让机制服务于不同 stage：Stage1 选择空间
感受野来生成首次 guided correction；Stage2/3 对已有 delta 做受限更新，优先使用
NAF 的 feature gating/channel scaling 来预测 a、b、gate。

### 3.2 保留的现有递推

```text
Stage1: [x_raw, x_coarse0, Sobel(x_coarse0)] -> delta1, gate1
x_coarse1 = x_coarse0 + delta1
Stage2: [x_raw, x_coarse1, Sobel(x_coarse1), delta1] -> a2, b2, gate2
delta2 = delta1 + gate2 * ((a2 * delta1 + b2) - delta1)
x_coarse2 = x_coarse0 + delta2
Stage3: [x_raw, x_coarse2, Sobel(x_coarse2), delta2] -> a3, b3, gate3
delta3 = delta2 + gate3 * ((a3 * delta2 + b3) - delta2)
final = x_coarse0 + delta3
```

Guided filter、输入输出、三个 stage 的递推公式、delta/a/b/gate 预测头和当前
edge-guided loss 均保留。只在各 stage 的 feature 到 prediction head 之间加入
stage-specific feature branch。

### 3.3 候选网络：Stage-specific residual feature branches

#### Stage1：VIV-inspired multi-scale correction

Stage1 的任务是根据 raw/coarse/edge 证据生成第一轮 delta。保留当前 stem 和
`delta_raw + gate_raw` head，在其隐藏 feature 后增加一个 BIM-inspired adapter：

```text
F = existing Stage1 feature
(F_local, F_context) = split(Conv1x1(F))
F_local   = DWConv3(F_local)
F_context = DWConv9(F_context)
M1: F_mix = Conv1x1(concat(F_local, F_context))
M2: compute VIV SW weights from channelwise cosine similarity,
    then fuse the independently reweighted branches
M3: apply VIV high-order interaction to the SW-weighted branches
deltaF = standard_init_1x1(F_mix)
F_new = F + alpha * deltaF       # alpha starts at 0
```

只将 `alpha` 初始化为 0，末层投影采用常规初始化；不要同时把二者置零，否则新支路
会出现乘积式零梯度。这样模型起始等价于原 Stage1，且 `alpha` 能在第一步获得梯度。
SW 按论文在两个分支间计算
channel-wise cosine similarity，MLP 输出每个通道的权重，按论文实现用 `1 + weight`
重标定两支；不改成互补的 softmax 权重。边缘条件已通过原有 Stage1 输入进入 F，
不另造一个论文中没有的 edge-attention。M1 先用简单 concat + 1x1 融合；M2 再加入
SW；HP 作为 M3 单独消融，测试论文配置 order 4，order 2 只作交互阶数/稳定性对照。
HP 仅作用于加权后的 local/context 两分支，结果经零初始化投影作为残差回到 F。

#### Stage2/3：NAF-inspired delta-update feature branch

Stage2/3 的目标是估计 delta 的仿射参数，而不是重新生成整幅图。因此保留原 stem
和 `a_raw, b_raw, gate_raw` head，在预测头前加入轻量 NAF-inspired feature unit：

```text
F = existing Stage2/3 feature
V = Conv1x1_expand(F) -> DWConv3(V)
V = SimpleGate(V)                   # split and elementwise multiply
V = SCA(V)                          # global channel reweight
deltaF = standard_init_1x1(V)
F_new = F + beta * deltaF       # beta starts at 0
```

Stage2 和 Stage3 的支路参数不共享，因为它们对应不同递推阶段。只将 `beta` 初始为
0，投影层采用常规初始化；不要再将投影同时零初始化，以免造成零梯度。`beta` 的
初始值使新支路在第一步等价旁路；
该分支只影响 a/b/gate 的特征，不直接添加图像 residual，不绕过 a/b 限幅和 gate。

#### 联合模型

只有两个分支单独有收益后，才组合为：

```text
guided filter
 -> original Stage1 + VIV-inspired multi-scale/SW branch (optional HP)
 -> original Stage2 + NAF-inspired SG/SCA feature branch
 -> original Stage3 + NAF-inspired SG/SCA feature branch
```

这保留我们方法主体，同时将 VIV 的感受野选择用于首次修正，将 NAF 的门控和通道
重标定用于 delta 更新。没有复制完整 U-Net、global residual 或其 RGB 训练设置。

### 3.4 实验顺序

| ID | 模型 | 变化 | 目的 |
|---|---|---|---|
| M0 | 当前 exp7C | 不改 | 三阶段结构基线 |
| M1-control | M0 + Stage1 双分支 control | 3x3/3x3 两分支，concat + 1x1 融合 | 对齐 VIV 的 Net(a) 基础消融 |
| M1 | M0 + Stage1 multi-scale | 3x3/9x9 两分支，concat + 1x1 融合，不用 SW/HP | 测试 9x9 分支的收益，并报告增加的参数/FLOPs |
| M1-budget | M0 + Stage1 matched-RF control | 3x3 + dilation-4 3x3 两分支 | 与 3x3/9x9 有近似参数/FLOPs，区分大感受野与密集大核采样 |
| M2 | M1 + Stage1 SW | 加入论文式 cosine similarity weighting | 相似性重标定是否带来额外收益 |
| M3 | M2 + Stage1 HP | 按论文 order 4 加高阶交互 | HP 是否进一步改善首次修正 |
| M4 | M0 + Stage2/3 SimpleGate | NAF-inspired gated feature residual，不含 SCA | 检验乘法门控对 delta 参数预测的作用 |
| M5 | M4 + Stage2/3 SCA | 再加轻量通道重标定 | 检验 SCA 的增量收益 |
| M6 | 最佳 M1-M3 + M5 | 合并已验证的两类分支 | 检查机制是否互补 |

每个 adapter 的零初始化残差系数使训练初始函数精确等于 M0。M1-control 对齐 VIV 论文 Net(a)；
M1-budget 用 dilation-4 3x3 作为近似参数/FLOPs 匹配的感受野控制。M2/M3 每次只增加
一个 VIV 机制。M4/M5 的 NAF 支路另设参数量匹配的普通卷积残差控制组。所有变体使用
相同训练协议、checkpoint 选择规则和验证集；test split 仅对锁定配置报告。

### 3.5 公平比较与验收

- 单通道输入与 GT；相同 train/val/test split、crop、有效 batch、optimizer updates、
  precision、scheduler 和随机种子策略。
- 第一轮沿用 exp7C loss，不同时引入 VIV 的 FFT loss 或新的 edge loss。
- 统一全图推理，报告 PSNR、MSE、MAE、edge-MAE、edge-MSE、halo score、参数量和
  推理时间。
- 模块需在验证集同时满足全图指标没有实质退化、edge 指标改善，才进入 M5；最终
  test 不参与结构或超参数选择。
- 当前 NAFNet 的 49.4338 dB 来自 60k checkpoint，exp7C 为 47.1141 dB；两者是
  强参照，但在确认 test IDs 和评估协议一致前，不把差值作为公平结构结论。

### 3.6 建议先实现的版本

先实现 M1-control/M1/M1-budget 和 M4/M5 两组独立变体，而不是一次做联合大模型：

1. M1-control、M1、M1-budget 保留 Stage1 stem/head，只添加初始残差缩放为零的双分支 adapter；
   随后在胜出分支上依次启用 SW、HP。
2. M4/M5 保留 Stage2/3 stem/head，先加初始残差缩放为零的 SimpleGate 支路，再加 SCA；
   与参数量匹配的普通卷积残差支路对照。
3. 仅当 Stage1 和 Stage2/3 分支各自有可重复验证集收益时，合并为 M6。
4. 所有代码和 checkpoint 放独立实验目录，原 exp7C 实现与权重保持只读基线。

这样第一轮直接检验两个清晰假设：VIV 的多感受野/相似性选择是否适合第一次 guided
correction；NAF 的乘法门控/通道重标定是否适合后续 delta 更新。
