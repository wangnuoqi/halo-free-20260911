# exp8B Clip-art 推理

这个目录用于调用 `20260815_edgepreserving` 中的 `exp8B_mixed_joint_fidelity` 完整模型。exp8B 使用 `mixed_lavel_singlehdr` 混合标签数据，可能包含多个 WLS 参数协议，不能把它标记为纯粹的 `(lambda=0.25, alpha=1.2)` 模型。虽然 checkpoint 文件名仍带有旧的 `l0p500_a1p400` 后缀，实际数据协议以 exp8B 训练脚本和数据目录为准。

## 模型级联

```text
JPEG degraded
  ├── 512×512 灰度 HR 输入
  └── 同一张 JPEG 下采样后的 LR 输入
          ↓
ConvGuided checkpoint
          ↓ coarse output
3-stage Refine checkpoint
          ↓
最终输出
```

exp8B 的 checkpoint 是按单通道数据训练的，因此脚本把 RGB JPEG 转成 YCbCr，只将亮度 Y 通道送入网络；输入 JPEG 的 Cb/Cr 色度通道保留并与预测亮度重新合成为彩色输出。clean PNG 只用于可选的参考指标，不进入网络输入。

## 运行

在 `halofree` 环境中：

```bash
cd /mnt/d/ResearchCode/halo-free/Decoupled_Learning/20261005_clipart_removal
python infer_exp7c_clipart/infer_exp7c_clipart.py
```

默认使用 `jpeg_q30`，输出到：

```text
output/exp8b_q30_10/
```

当前三联输出模式要求对应的 `clean_png` 存在；如果缺少 clean GT，该样本会被跳过，以保证每个输出样本都同时包含 input、GT 和 model output。

当前项目要求的三种图像会保存在同一个输出目录中，每个样本有三个文件：

```text
*_input.jpg          # 原始 degraded JPEG 输入
*_clean_gt.png       # clean PNG GT
*_exp8b_output.png   # exp8B 真实模型输出
```

测试 `Q=50`：

```bash
python infer_exp7c_clipart/infer_exp7c_clipart.py \
  --input-dir openclipart_500/jpeg_pairs_512/jpeg_q50 \
  --output-dir output/exp8b_q50_10
```

先只跑 5 张检查输入协议和输出：

```bash
python infer_exp7c_clipart/infer_exp7c_clipart.py --max-images 5
```

输出包括：

- `*_exp8b_output.png`：exp8B 最终彩色输出；
- `*_input.jpg`：原始 degraded JPEG 输入；
- `*_clean_gt.png`：对应 clean GT；
- `metrics.csv`：逐图输入/输出 PSNR、MAE；
- `metrics.txt`：平均指标。

## 注意

- 必须同时加载 ConvGuided 和 Refine 两个 checkpoint；
- 不要把 clean PNG 输入网络；它只用于可选评估；
- `lr` 是 degraded JPEG 的下采样版本，不能从 clean PNG 生成；
- 当前脚本只运行 exp8B，不运行传统 WLS；
- 当前脚本使用 YCbCr 的 Y 通道作为模型输入，并保留 degraded JPEG 的 Cb/Cr 通道生成彩色可视化；
- 现有 exp8B 训练源码含旧的 Windows/WSL 数据路径，本脚本只抽取其中的模型定义，不会执行训练或读取旧数据集。
