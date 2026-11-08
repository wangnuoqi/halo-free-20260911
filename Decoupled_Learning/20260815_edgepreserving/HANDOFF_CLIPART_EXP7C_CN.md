# Clip-art 外部应用实验 Handoff：exp7C

## 当前确定的模型

后续在 Openclipart JPEG 数据上进行 V8/WLS 保边和伪影抑制测试时，固定使用：

```text
exp7C_20260818_joint_fidelity
```

exp7C 是当前选定的完整 joint-training pipeline。它同时优化第一阶段 ConvGuided 和后面的三阶段 Refine，并额外加入 coarse fidelity loss，避免第一阶段输出在 joint training 中严重退化。

## 对应代码

训练源代码：

```text
experiments/exp7C_joint_fidelity_20260818.py
```

启动脚本：

```text
experiments/exp7C_run.ps1
```

训练配置的关键部分：

```text
数据：20260818 WLS 数据
WLS 参数：lambda=0.500，alpha=1.400
训练轮数：80
Refine 学习率：3e-3
ConvGuided 学习率：3e-4
联合训练：是
coarse fidelity loss：0.5 * SmoothL1(coarse, GT)
```

## 必须同时加载两个 checkpoint

exp7C 不是一个只有 Refine 的模型。完整推理流程是：

```text
输入图像
    ↓ ConvGuided checkpoint
coarse output
    ↓ 3-stage Refine checkpoint
最终输出
```

因此后续调用时必须同时使用下面两个文件。

### ConvGuided checkpoint

```text
experiments/exp7/exp7C_20260818_joint_fidelity/models/exp7C_20260818_joint_fidelity/convguided_joint_best_20260818_l0p500_a1p400.pth
```

### Refine checkpoint

```text
experiments/exp7/exp7C_20260818_joint_fidelity/models/exp7C_20260818_joint_fidelity/rf_3stage_best_20260818_l0p500_a1p400.pth
```

不能只加载 Refine checkpoint 并使用原来的 ConvGuided checkpoint，因为 exp7C 在训练过程中已经联合更新了 ConvGuided。两个 checkpoint 必须成对使用。

## 为什么选择 exp7C

### exp6E

`exp6E` 是最早明确的 ConvGuided + Refine joint-training 实验，但它没有 coarse fidelity 约束。其 joint training 后的 ConvGuided-only 输出明显退化，Refine 在补偿第一阶段的退化。

### exp7C

exp7C 在 exp6E 的 joint training 基础上加入：

```python
loss = loss + 0.5 * F.smooth_l1_loss(batch_input, batch_label)
```

这项损失约束 ConvGuided coarse output 保持与 GT 的一致性，因此 exp7C 比 exp6E 更适合作为完整外部应用模型。

已有独立评估结果：

```text
ConvGuided MSE：0.00044515
最终 RF MSE：0.00002273
最终 RF PSNR：47.1141 dB
```

### exp8A / exp8B

exp8A 和 exp8B 也是 joint fidelity 版本，但它们是后续针对其他数据集重新训练的实验。当前 clip-art 外部应用第一轮固定使用 exp7C，避免混入不同数据协议。

## Clip-art 测试数据

数据生成目录：

```text
/mnt/d/ResearchCode/halo-free/Decoupled_Learning/20261005_clipart_removal/openclipart_500/jpeg_pairs_512
```

目录中包含：

```text
clean_png/   # SVG 渲染得到的 512×512 clean reference
jpeg_q10/    # JPEG quality=10
jpeg_q20/    # JPEG quality=20
jpeg_q30/    # JPEG quality=30
jpeg_q50/    # JPEG quality=50
```

当前数据统计：

```text
成功生成的 clean 图：475 张
每个 JPEG quality 目录：475 张
质量因子：10、20、30、50
色度子采样：4:2:0
```

数据生成逻辑和限制见：

```text
/mnt/d/ResearchCode/halo-free/Decoupled_Learning/20261005_clipart_removal/openclipart_500/jpeg_pairs_512/DATA_HANDOFF_CN.md
```

## 第一轮实验协议

先使用 `jpeg_q30` 和 `jpeg_q50`，不要一开始就使用最严重的 `jpeg_q10`。

对每张 JPEG 输入执行：

```text
JPEG clip-art
    ├── exp7C ConvGuided + Refine
    └── 与 exp7C 对应的固定参数 WLS
```

比较两类结果：

### 1. 算子和网络行为

```text
exp7C output vs WLS output
```

观察：

- 输出差异；
- 边缘位置和梯度一致性；
- 平坦区域变化；
- block/ringing 抑制是否相似。

### 2. 相对于 clean reference 的应用效果

```text
JPEG input vs clean_png
WLS output vs clean_png
exp7C output vs clean_png
```

可计算：

- PSNR；
- SSIM；
- LPIPS；
- 边缘区域误差；
- 平坦区域误差。

`clean_png` 是由同一 SVG 渲染得到的实验参考，不是 exp7C 的训练 target，也不是 WLS 算子的 target。

## 第一轮不做的事情

当前阶段不要：

- 用 clip-art 数据重新训练 exp7C；
- 把 exp7C 重新定义为 JPEG 专用 restoration 网络；
- 加入 ARCNN、FBCNN 或其他 JPEG 网络作为比较；
- 把 Office-Home 已有 JPG 直接当作 clean GT；
- 只加载 Refine checkpoint 而忽略 joint ConvGuided checkpoint；
- 在测试集上重新调整 WLS 参数。

当前目标是测试：

> 已经在 WLS 任务上联合训练的 exp7C pipeline，能否直接迁移到 clip-art JPEG 输入，并保持保边和伪影抑制行为。

## 后续 3-stage refine 方向

只有在 exp7C 直接测试表现出明确问题后，才考虑额外的 refine 结构。需要先区分问题来源：

```text
exp7C 与 WLS 差异很大
    → 先检查输入范围、颜色空间和预处理

exp7C ≈ WLS，但 WLS 本身对 JPEG 效果差
    → 说明固定 WLS 先验可能不适合 JPEG

exp7C 和 WLS 都有改善，但细线/边缘损失明显
    → 再研究 3-stage refine 是否能修正过平滑
```

不要在第一轮就加入新的 refine 模块，否则无法判断问题来自 exp7C 本身、WLS 算子还是新增结构。

## 路径注意事项

部分训练脚本内部仍保留 Windows 路径和旧的 WSL 路径配置，例如：

```text
D:\\nuoqi\\halo-free\\Decoupled_Learning\\20260815_edgepreserving
/mnt/d/nuoqi/halo-free/Decoupled_Learning/20260815_edgepreserving
```

当前实际代码目录是：

```text
/mnt/d/ResearchCode/halo-free/Decoupled_Learning/20260815_edgepreserving
```

因此后续编写 clip-art 推理脚本时，建议显式传入 checkpoint 路径，不要直接依赖训练脚本内部的旧路径。

## 一句话调用说明

后续如果需要调用当前确定的网络，应说：

> 使用 `20260815_edgepreserving` 中的 `exp7C_20260818_joint_fidelity`，同时加载 `convguided_joint_best_20260818_l0p500_a1p400.pth` 和 `rf_3stage_best_20260818_l0p500_a1p400.pth`，在 `jpeg_pairs_512/jpeg_q30` 或 `jpeg_q50` 上做 clip-art 测试。
