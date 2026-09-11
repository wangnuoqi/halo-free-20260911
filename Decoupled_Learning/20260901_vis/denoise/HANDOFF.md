# Denoise 可视化交接说明

## 目的

本目录用于比较 BSD68 上的三个 denoise 输出：

1. 我们的 `Exp8A ConvGuided + frozen Exp6D RF`；
2. 官方 `FFDNet sigma25`；
3. 官方 `DnCNN sigma25`。

可视化代码全部位于 `20260901_vis/denoise`，不修改
`20260901_vis/edge` 下的任何文件。

## 输入来源

可视化脚本不重新加载模型，也不重新生成噪声，而是读取已经按同一 BSD68 协议生成的推理结果：

```text
outputs/exp8a_patch_frozen/
  final/     # Exp8A ConvGuided + RF 输出
  gt/        # BSD68 clean GT

outputs/official_sigma25/
  ffdnet_official_sigma25/coarse/  # 官方 FFDNet 输出
  dncnn_official_sigma25/coarse/   # 官方 DnCNN 输出
```

三组输出来自同一个 `/mnt/d/nuoqi/halo-free/data_bsds/BSD68`，共 68 张图，使用同一套 sigma25 AWGN 和 `RandomState(0)` 噪声协议。`gt` 是 BSD68 clean 原图。

## 输出内容

运行后默认生成：

```text
visualization/exp8a_vs_official/
  four_panel/
    test001_four_panel.png       # ConvGuided+RF | FFDNet | DnCNN | GT
  individual/
    test001_ours.png             # 单独放大的 ConvGuided+RF
    test001_ffdnet.png           # 单独放大的 FFDNet
    test001_dncnn.png            # 单独放大的 DnCNN
    test001_gt.png               # 单独放大的 GT
  error_heatmaps/
    test001_error_heatmap.png    # 三个模型相对 GT 的绝对误差热图
  edge_analysis/
    summary/                    # 单张保边总览图：输出、GT 和边缘保持叠加
      individual/               # 下排每个子图的独立高清版本
        test001_ours_edge_retention.png
        test001_ffdnet_edge_retention.png
        test001_dncnn_edge_retention.png
        test001_gt_strong_edge_reference.png
  per_image_metrics.csv          # 每张图的 PSNR 和 MSE
  normalization.txt              # 本次可视化的归一化记录
```

四联图用于直接观察视觉差异。单图全部平铺在同一个 `individual/` 目录中，文件名后缀区分模型和图类型。

像素误差热图不是模型输出，而是：

```text
abs(output - GT)
```

GT 不单独绘制误差图，因为它相对自身的误差恒为零。

## 保边可视化

保边分析使用 GT 的 Sobel 梯度作为统一参考：

```text
gradient_error = sqrt((Gx(output)-Gx(GT))^2 + (Gy(output)-Gy(GT))^2)
edge_ratio = Sobel(output) / Sobel(GT)
```

GT 强边缘 mask 取每张 GT Sobel 幅值图高于第 90 百分位的最强前 10% 像素，并扩张 1 个像素形成 edge band。三种模型使用同一张 GT mask，不根据各自输出重新选边缘。

`edge_ratio` 只在 GT 强边缘上显示，色标固定为 `[0.5, 1.5]`：接近 `1` 表示边缘强度保持，低于 `0.85` 表示边缘变弱，高于 `1.15` 表示可能过锐化或放大噪声。

主推荐图是 `edge_analysis/summary/test001_edge_preservation_summary.png`。它是一张 2x4 总览图：上排为三个模型输出和 GT；下排为三个模型输出叠加 GT 强边缘上的边缘强度比值，以及 GT 边缘参考。下排颜色编码为蓝色（边缘变弱）、白色（边缘保持）、红色（边缘变强），因此一张图可以同时看实际去噪结果和保边情况。

下排的四个子图也会独立保存到 `edge_analysis/summary/individual/`，文件名使用 `_ours_edge_retention`、`_ffdnet_edge_retention`、`_dncnn_edge_retention` 和 `_gt_strong_edge_reference` 后缀，便于放大比较。

`per_image_metrics.csv` 还记录每种方法的 edge-band gradient MAE、梯度保持比例、弱边缘比例和强边缘比例。这些指标比普通像素误差热图更适合判断保边效果。为避免输出过于分散，`edge_analysis/` 只保留 `summary/` 及其 `individual/` 子目录，不再生成单独的 Sobel、梯度误差和边缘比值目录。

## 归一化规则

强度图的显示范围始终固定为 `[0, 1]`，三个模型和 GT 不进行各自的 min-max 拉伸。

误差热图的下限固定为 `0`。上限默认在本次选中的所有样本、所有三个模型的误差上统一计算：取每张图三种误差合并后的 `99.5` 百分位，再取这些百分位中的最大值。这个单一上限用于所有热图；超过上限的极少数像素只在显示时截断，不改变原始输出和指标。具体数值记录在 `normalization.txt`。

边缘比值叠加图固定使用 `[0.5, 1.5]`，不做逐图归一化。

如果需要多次运行时使用完全固定的色标，可以显式指定：

```bash
ERROR_VMAX=0.15 bash run_denoise_visualization.sh
```

## 运行方法

生成全部 68 张图：

```bash
bash /mnt/d/nuoqi/halo-free/Decoupled_Learning/20260901_vis/denoise/run_denoise_visualization.sh
```

只生成指定样本，便于快速检查：

```bash
SAMPLES=test001,test020 bash /mnt/d/nuoqi/halo-free/Decoupled_Learning/20260901_vis/denoise/run_denoise_visualization.sh
```

这个脚本只做可视化，不会覆盖 `exp8a_patch_frozen` 或 `official_sigma25` 的推理结果。
