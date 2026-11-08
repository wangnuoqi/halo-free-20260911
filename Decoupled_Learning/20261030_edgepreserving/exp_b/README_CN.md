# Exp B：只替换 LRNet 内部 block

## 实验问题

在保留现有保边滤波流程的前提下，只把原始 LR predictor 中四个 dilation block 的内部变换替换成 VIV-inspired block，能否提升或至少保持最终效果？

本实验不加载 Exp A、C0/exp7C、NAFNet 或 VIVNet 的权重。所有权重从随机初始化开始，按 base → refine → joint 顺序训练。

## 模型结构边界

- LR predictor：单通道输入、24 通道中间特征、四个 dilation 位置（2、4、8、16）、单通道 raw 输出；不添加 sigmoid。
- ConvGuided：保持 box statistics、局部仿射系数估计、双线性上采样和 [0,1] clamp。
- Refine：保持三个逐像素 stage 的 delta / a / b / gate 递推。
- LR 与 HR 在调用模型前各减 0.5 一次；ConvGuided 输出回到 [0,1] 后，送入 Refine 前再减 0.5。

只替换 LR predictor 中四个 24→24 dilation 卷积 block。新 block 包含 3×3 depthwise 分支、9×9 depthwise 分支、channel similarity weighting、逐元素高阶乘法交互和 1×1 projection。projection 零初始化，因此 block 初始时近似原模型的 identity-style 初始化和 LeakyReLU 输出。

## 数据和训练协议

数据根目录：/mnt/d/ResearchCode/S2R-HDR/data/20260818

数据读取器按文件名 stem 配对 LR、HR 和 GT；GT 文件名中的 -WLS-l0p500-a1p400 会映射到输入文件对应的 -log stem。已核对目录中有 2003 组训练三元组和 200 组测试三元组，HR/LR 空间倍率为 2。

训练时按 seed=42 从 2003 组训练数据中固定留出 200 组验证图，剩余 1803 组用于训练。训练使用成对随机 crop：HR/GT 为 128×128，LR 为 64×64；crop 起点按 2 像素网格对齐。验证使用留出图中心的 128×128 paired crop。独立 200 张测试集只用于最后汇报，不参与选 best checkpoint。

默认训练设置：

| 项目 | 设置 |
|---|---|
| 分布式方式 | 4 卡 DDP + NCCL |
| 每卡 batch | 16 |
| 全局 batch | 64 |
| patch | HR/GT 128×128；LR 64×64 |
| 精度 | BF16 |
| optimizer | AdamW，weight decay 1e-4 |
| 学习率 | 3e-4，cosine 到 3e-6 |
| DataLoader workers | 每个 rank 默认 4，可用 NUM_WORKERS=8 等覆盖 |
| pinned memory | 关闭，pin_memory=False |
| seed | 42 |

每个阶段默认 200 epochs。三阶段默认使用相同的数据划分、patch、batch、精度和 optimizer 配置；阶段学习率可以通过 LR 环境变量统一调整。

## 三阶段训练目标

### 1. Base

训练 VIV-block LR predictor 和 ConvGuided，Refine 不参与优化：

    MSE(coarse0, GT)
    + 0.1 × 0.5 × (L1(SobelX(coarse0), SobelX(GT))
                  + L1(SobelY(coarse0), SobelY(GT)))

按验证 crop 的逐图 MSE 选择 base_best.pth。

### 2. Refine

从 base_best.pth 加载 base 并冻结它，只训练原始三阶段 Refine。使用 exp7C 对齐的 delta-space cascade loss：

    0.1 × edge_loss(delta1, GT - coarse0)
    + 0.3 × edge_loss(delta2, GT - coarse0)
    + 0.6 × edge_loss(delta3, GT - coarse0)

edge mask 从 target delta 的 Sobel magnitude 用逐图 Otsu 计算，loss 包括平滑区像素项、边缘像素项和边缘梯度项。按验证 crop 的最终输出逐图 MSE 选择 refine_best.pth。

### 3. Joint

同时加载 base_best.pth 与从该 base 训练得到的 refine_best.pth，解冻全模型联合优化：

    delta-space cascade loss + 0.5 × SmoothL1(coarse0, GT)

额外的 coarse fidelity 项沿用 exp7C，避免联合训练时 coarse0 大幅退化。按验证 crop 最终输出逐图 MSE 选择 joint_best.pth。

三个阶段均保存 optimizer、scheduler、epoch、step、协议和验证指标；每个 epoch 更新 *_latest.pth，验证指标改善时更新 *_best.pth，正常结束时保存 *_final.pth。训练历史写入 loss_history.jsonl。恢复训练时，除 workers 数外，阶段、数据、patch、batch、world size、精度、学习率和 epoch 数必须与 checkpoint 协议一致。

## 启动训练

从项目目录运行：

    cd /mnt/d/ResearchCode/halo-free/Decoupled_Learning/20261030_edgepreserving

    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    CROP_SIZE=128 \
    BATCH_SIZE_PER_GPU=16 \
    EPOCHS=200 \
    LR=3e-4 \
    MIN_LR=3e-6 \
    NUM_WORKERS=4 \
    PRECISION=bf16 \
    PYTHON_BIN=/home/nuoqi/miniconda3/envs/halofree/bin/python \
    bash exp_b/run_sequence_ddp.sh

需要保留更多 CPU workers 时，可把 NUM_WORKERS=4 改成 NUM_WORKERS=8。每个阶段结束后脚本才会启动下一个阶段。

默认输出根目录：

    exp_b/runs/exp_b_vivblock_20261102/
    ├── base/
    ├── refine/
    └── joint/

若某阶段中断，单独恢复该阶段，例如：

    START_PHASE=refine RESUME=1 NUM_WORKERS=8 \
    PYTHON_BIN=/home/nuoqi/miniconda3/envs/halofree/bin/python \
    bash exp_b/run_sequence_ddp.sh

START_PHASE 可为 base、refine 或 joint。恢复时 RESUME=1 只作用于指定起始阶段；后续阶段按新阶段正常启动。也可以分别运行 run_base_ddp.sh、run_refine_ddp.sh、run_joint_ddp.sh。

## 测试集评估

训练完成后在 200 张测试图上进行全图推理，底部或右侧奇数边裁掉；不保存 prediction，只输出逐图和汇总指标：

    cd /mnt/d/ResearchCode/halo-free/Decoupled_Learning/20261030_edgepreserving
    CUDA_VISIBLE_DEVICES=1 /home/nuoqi/miniconda3/envs/halofree/bin/python -u \
      exp_b/src/evaluate_exp_b.py \
      --data-root /mnt/d/ResearchCode/S2R-HDR/data/20260818 \
      --base-checkpoint exp_b/runs/exp_b_vivblock_20261102/base/base_best.pth \
      --refine-checkpoint exp_b/runs/exp_b_vivblock_20261102/refine/refine_best.pth \
      --joint-checkpoint exp_b/runs/exp_b_vivblock_20261102/joint/joint_best.pth \
      --output-dir exp_b/evaluation/test_20260818

可选地在同一汇总中加入 exp7C、NAFNet、VIVNet 的历史 summary.json：

    --reference-summary "exp7C=/mnt/d/ResearchCode/halo-free/Decoupled_Learning/20260815_edgepreserving/baseline/evaluation/exp7c_test/summary.json" \
    --reference-summary "NAFNet=/mnt/d/ResearchCode/halo-free/Decoupled_Learning/20260815_edgepreserving/baseline/evaluation/nafnet_wls_test/summary.json" \
    --reference-summary "VIVNet=/mnt/d/ResearchCode/halo-free/Decoupled_Learning/20260815_edgepreserving/baseline/evaluation/vivnet_wls_test/summary.json"

评估目录只写 summary.json 和 per_image_metrics.csv。Exp B 三个阶段会记录 coarse0；Refine 和 Joint 还记录 coarse1、coarse2 和 final。

### 2026-11-04 独立 test-200 结果

已使用 Windows `yolo_clean` CUDA 环境、单卡 FP32，在 200 张独立测试图上对 base、refine、joint 的 best checkpoint 做全图评估；图像底部/右侧奇数边裁掉，不保存预测图。汇总和逐图结果分别为 `evaluation/test_20260818/summary.json` 与 `evaluation/test_20260818/per_image_metrics.csv`。

| Checkpoint | 输出 | Mean per-image PSNR (dB) | Mean per-image MSE |
|---|---|---:|---:|
| Base best | coarse0 | 38.309 | 1.632e-4 |
| Refine best | coarse0 | 38.309 | 1.632e-4 |
| Refine best | final | 36.030 | 2.671e-4 |
| Joint best | coarse0 | 37.877 | 1.845e-4 |
| Joint best | coarse1 | 38.673 | 1.465e-4 |
| Joint best | coarse2 | 38.210 | 1.674e-4 |
| Joint best | final | 37.725 | 2.002e-4 |
| Exp7C reference | final | 47.114 | 2.273e-5 |
| VIVNet reference | final | 48.992 | 1.620e-5 |
| NAFNet reference | final | 49.434 | 1.476e-5 |

关键发现：joint best 的独立测试集 final 为 37.725 dB，比其 coarse0 的 37.877 dB 还低；refine best 的 final 为 36.030 dB，也低于 base 的 38.309 dB。训练验证中心 crop 上 joint best 约 41.5 dB，显著高于独立 test-200 的 37.725 dB，说明验证表现没有泛化到测试集。当前 Exp B 不能作为质量提升候选；后续应先分析 refine 在 test-200 上造成的退化，并核对 train/validation/test 的输入或目标分布，再决定是否保留三阶段 refine 或继续调结构。此结果也不能单独归因于 VIV-inspired block，因为尚无同协议的原始 LR block 训练对照。

## 当前状态

- 模型、数据、训练、评估入口和四卡启动脚本已经实现。
- 已用真实 20260818 数据核对到 2003 个 train 三元组和 200 个 test 三元组。
- CPU 单步 smoke test 已分别通过 base、refine、joint，可保存三种阶段 checkpoint。
- 四卡 RTX 3090 DDP/NCCL 的 base 单步 smoke test 已通过，包含训练、验证和 checkpoint 保存；该 smoke 使用 batch/GPU=1、val_count=2，只验证 NCCL 训练链路，不是正式训练结果。
- 正式训练已完成；独立 test-200 全图结果见上表。测试表明当前 refine/joint 泛化较差，Exp B 尚未达到 Exp7C、NAFNet 或 VIVNet 的质量。
