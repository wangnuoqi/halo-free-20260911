# RGBDSR 深度图可视化

这里保存一份独立复制的 x8 checkpoint 和推理代码，用于后续稳定复用、生成模型横向对比图。当前只保留 ConvGuided、DKN、DCTNet 及原始 ConvGuided RF。

## 运行

```bash
cd /mnt/d/nuoqi/halo-free/Decoupled_Learning/20260901_vis/rgbdsr
bash run_rgbdsr_visualization.sh --models conv_rf dkn dctnet gt --ids 0 17 42
```

省略 `--ids` 会处理 449 张测试图。GPU 可通过 `CUDA_VISIBLE_DEVICES=0` 指定；没有 CUDA 时自动使用 CPU。

## 输出

`outputs/x8/` 下：

- `depth_panels/{id}.png`：四联总图；`depth_panels/individual/` 保存带模型后缀的单独子图。
- `depth_error_panels/{id}.png`：三个模型相对 GT 的毫米绝对误差总图；`individual/` 保存单独子图。统一色标为 `0--100 mm`，可通过 `--error-max-mm` 修改。
- `gradient_panels/{id}.png`：四联 Sobel 梯度幅值总图；`gradient_panels/individual/` 保存单独子图。
- `gradient_error_panels/{id}.png`：三个模型梯度差异总图；`gradient_error_panels/individual/` 保存单独子图。
- `run_manifest.json`：数据、倍率、色标和模型清单。

评估指标按 x8、crop=6、逐图 RMSE 宏平均协议单独计算；当前可视化只保存图像和指标，不保存中间数组。
