# CRA/HiFill 官方代码运行记录

- 官方仓库：`Atlas200dk/sample-imageinpainting-HiFill`
- GPU/CPU 实现：`GPU_CPU/test.py`
- 预训练模型：`GPU_CPU/pb/hifill.pb`（约 11 MB）
- 运行环境：临时 `/tmp/hifill-venv`，Python 3.9，TensorFlow CPU 2.10.1（通过 `tf.compat.v1` 读取 TF1 frozen graph），OpenCV 4.8.1。
- 运行命令：在 `GPU_CPU` 目录执行兼容 TensorFlow wrapper，官方样例 8 张图均成功输出到 `GPU_CPU/results/`。

官方代码中的真实后处理逻辑：

1. 将输入缩放到 `multiple * 512`，再平均下采样到 512×512；
2. frozen graph 只在 512×512 上预测 `inpainted` 和 attention；
3. 将 512 结果双线性放大为 `low_base`；
4. 计算高分辨率 `residual = (large_img - low_large) * large_mask`；
5. 用 attention 对 mask 外 residual patch 做加权聚合；
6. `res_large = low_base + aggregated_residual`，mask 外复制原图。

因此，论文所说的 residual aggregation 在官方实现中确实存在；它不是简单把低分辨率结果插值后直接作为最终结果。

当前尚未做严格的论文 ablation（同一输入下关闭 residual aggregation 与开启 CRA 的像素/边界指标对比），因为官方脚本没有保存中间 `low_base`。但从输出和代码可以确认该实验可以直接加入：保存 `low_base` 与 `res_large`，然后比较 mask 内清晰度、梯度和 halo。
