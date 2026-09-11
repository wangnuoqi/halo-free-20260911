# 20260705_vis

这个目录主要用于可视化和分析两阶段推理流程：

1. 先用 `eva_convguided_base.py` 跑基础的 Deep Guided Filter 模型，生成第一阶段输出。
2. 再用 `eva_refine_3stage.py` 读取第一阶段输出，跑三级细化模型，生成第二阶段输出。
3. 最后用 `analyze_model2_vs_label.py` 将第二阶段输出与 GT label 对比，导出误差统计和可视化结果。

这里同时保存了若干现成的输出目录，用于结果检查和论文图整理。

## 目录中的主要代码

### `eva_convguided_base.py`

作用：对基础 `DeepGuidedFilter` 模型做推理，并把结果保存成 `float32 TIFF`。

它做的事情包括：

- 从低分辨率输入目录和高分辨率输入目录中按文件名配对样本。
- 在脚本内部定义 `DeepGuidedFilter`、`ConvGuidedFilter`、`AdaptiveNorm` 等推理所需网络结构。
- 加载 `convguided_filter_best_mix.pth` 权重。
- 对输入做对齐裁剪，保证高宽为偶数。
- 输出第一阶段预测结果到指定目录，默认是 `model_outputs/`。
- 打印平均单张推理耗时。

默认输入输出：

- LR 输入：`/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/down_sample_2_testinput`
- HR 输入：`/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/testinput`
- 模型权重：`/mnt/d/nuoqi/halo-free/Decoupled_Learning/total-pipeline-v1/checkpoints/convguided_filter_best_mix.pth`
- 输出目录：`/mnt/d/nuoqi/halo-free/Decoupled_Learning/20260705_vis/model_outputs`

示例：

```bash
python eva_convguided_base.py
```

也可以显式指定参数：

```bash
python eva_convguided_base.py \
  --lr-dir /mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/down_sample_2_testinput \
  --hr-dir /mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/testinput \
  --model-path /mnt/d/nuoqi/halo-free/Decoupled_Learning/total-pipeline-v1/checkpoints/convguided_filter_best_mix.pth \
  --output-dir /mnt/d/nuoqi/halo-free/Decoupled_Learning/20260705_vis/model_outputs
```

### `eva_refine_3stage.py`

作用：对第一阶段结果做三级级联细化，输出最终预测结果，并顺带统计 PSNR、MSE、loss 和推理时间。

核心结构：

- `Stage1DeltaNet`：先预测一个初始残差 `delta1`。
- `ABAdjustNet`：预测线性修正参数 `a, b`，将上一阶段的残差变换成下一阶段残差。
- `CascadeRefine3Stage`：把 3 个阶段串起来，依次得到 `delta1 -> delta2 -> delta3 -> final`。

脚本里还实现了：

- Sobel 梯度计算。
- Otsu/百分位/固定阈值的边缘掩码构造。
- 带边缘约束、梯度约束、低频约束的 `edge_guided_loss`。
- 针对三级残差监督的 `cascade_loss`。

当前脚本用途更偏向“评估式推理”而不是通用 CLI：

- 读取原始输入：`testinput`
- 读取第一阶段输出：`model_outputs_model1`
- 读取 GT：`testlabel`
- 加载三阶段细化权重：`rf_3stage_best_otsu_v3.pth`
- 输出最终结果到：`model_outputs_model2`

默认路径：

- 原始输入：`/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/testinput`
- 第一阶段输出：`/mnt/d/nuoqi/halo-free/Decoupled_Learning/20260705_vis/model_outputs_model1`
- GT：`/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/testlabel`
- 模型权重：`/mnt/d/nuoqi/halo-free/Decoupled_Learning/total-pipeline-v1/checkpoints/rf_3stage_best_otsu_v3.pth`
- 输出目录：`/mnt/d/nuoqi/halo-free/Decoupled_Learning/20260705_vis/model_outputs_model2`

运行方式：

```bash
python eva_refine_3stage.py
```

说明：

- 这个脚本目前没有做成命令行参数形式，路径都写在文件内部。
- 如果要复用到别的数据集，通常需要先改脚本顶部的几个目录变量和 `model_path`。

### `analyze_model2_vs_label.py`

作用：把第二阶段预测结果和 GT label 对齐后逐张分析，输出误差图、结构叠加图、逐图统计以及整体汇总。

主要分析内容：

- 基础误差指标：`MAE`、`RMSE`、`max_abs_error`、`p90/p99_abs_error`
- 相对误差指标：平均、median、p90、max relative error
- 符号误差：`signed_error_mean`
- 结构区域统计：基于 label 的 Sobel 梯度，把图像分成 `strong / weak / flat` 三类区域，并统计各区域中的误差分布

主要输出：

- `per_image_vis/`：每张图的四联误差图
- `per_image_structure/`：结构区域和 signed error 叠加图
- `per_image_stats/`：每张图的文本统计
- `summary.csv`：逐图汇总表
- `summary.txt`：整体统计摘要

默认输入输出：

- 预测目录：`/mnt/d/nuoqi/halo-free/Decoupled_Learning/20260705_vis/model_outputs_model2`
- GT 目录：`/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/testlabel`
- 输出目录：`/mnt/d/nuoqi/halo-free/Decoupled_Learning/20260705_vis/error_analysis_model2`

示例：

```bash
python analyze_model2_vs_label.py
```

或：

```bash
python analyze_model2_vs_label.py \
  --pred-dir /mnt/d/nuoqi/halo-free/Decoupled_Learning/20260705_vis/model_outputs_model2 \
  --label-dir /mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/testlabel \
  --output-dir /mnt/d/nuoqi/halo-free/Decoupled_Learning/20260705_vis/error_analysis_model2
```

## 结果目录说明

### `model_outputs_model1/`

第一阶段模型输出。供三阶段细化脚本继续读取。

### `model_outputs_model2/`

三阶段细化后的最终输出。供误差分析脚本读取。

### `error_analysis_model2/`

`analyze_model2_vs_label.py` 的一套完整分析结果，包含：

- 每张图的误差可视化
- 每张图的结构叠加图
- 每张图的统计文本
- 全部样本的汇总表和摘要

### `finalerror_analysis_model2/`

看目录结构和命名，这也是一套“最终版”的 model2 误差分析结果缓存，内容形式与 `error_analysis_model2/` 相同，主要是另一轮整理后的导出结果。

### `error_analysis_model2_top10_p90/`

从 `error_analysis_model2/` 中额外筛出的高误差样本子集。当前目录里保存了：

- `top10_list.csv`
- 这 10 张图对应的 `per_image_vis/`
- 这 10 张图对应的 `per_image_stats/`

从命名看，它用于重点查看 `p90` 误差较高的样本。

### `vis_comp/`

多方法可视化对比素材目录。包含：

- `DualCNN_2022/`
- `Fan_2018/`
- `Xu_2015/`
- `Ours_wo_refine_stage/`
- `Our_full_model/`

这些子目录里存的是不同方法在同一批样例上的预测结果，通常用于做论文图或主观可视化对比。

## 推荐使用顺序

如果你要重新走一遍流程，建议顺序是：

```text
eva_convguided_base.py
  -> 生成第一阶段输出

eva_refine_3stage.py
  -> 读取第一阶段输出，生成最终输出

analyze_model2_vs_label.py
  -> 对最终输出做误差分析与可视化
```

## 依赖

脚本中实际使用到的主要依赖包括：

- Python 3
- PyTorch
- NumPy
- OpenCV
- imageio
- matplotlib
- tqdm

其中：

- `eva_convguided_base.py`、`eva_refine_3stage.py` 依赖 PyTorch
- `analyze_model2_vs_label.py` 依赖 `matplotlib` 生成误差图

## 备注

- 这个目录里还有测试脚本，但本 README 不覆盖测试代码。
- `eva_refine_3stage.py` 当前是实验脚本风格，路径硬编码较多；如果后面要长期复用，建议再补一个命令行参数版本。
