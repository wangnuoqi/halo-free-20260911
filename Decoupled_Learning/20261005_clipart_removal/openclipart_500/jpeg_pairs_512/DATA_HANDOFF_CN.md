# Clip-art JPEG 去伪影数据生成说明

## 目的

这套数据用于测试已有的 V8 WLS 模拟网络，以及与它使用相同 WLS 参数的传统 WLS 算子，在 clip-art 图像上的保边和 JPEG 伪影抑制效果。

当前实验不针对 clip-art 重新训练 V8。数据中的 clean PNG 主要用于：

- 作为 JPEG 压缩前的参考图像；
- 计算 PSNR、SSIM、LPIPS 等 clean-reference 指标；
- 检查 WLS/V8 是否在去除伪影的同时损失细线、轮廓和颜色边界。

## 数据来源

原始图形来自 Openclipart SVG 镜像：

- 仓库：[cyanidecupcake/openclipart-svg](https://github.com/cyanidecupcake/openclipart-svg)
- 原始目录：`openclipart_500/svg_raw/`

SVG 是矢量源文件，不包含 JPEG 编码伪影。下载脚本和随机抽样信息记录在上一级目录的 `manifest.json` 中。

## 生成流程

每个样本按照下面的流程生成：

```text
SVG 矢量图
    ↓ CairoSVG 渲染
512×512 RGB clean PNG
    ↓ Pillow JPEG 编码
JPEG degraded 图像
```

具体处理如下：

1. 读取 SVG 文件。
2. 使用 CairoSVG 将 SVG 渲染到正方形画布。
3. 保持原始图形的宽高比，不拉伸图形；图形居中放置。
4. 透明区域统一合成到白色背景。
5. 输出 `512×512` 的 RGB PNG，作为 clean reference。
6. 使用 Pillow 按不同 JPEG quality factor 保存 JPEG。
7. JPEG 输入与对应 PNG 使用同一张 SVG 生成，因此二者内容和像素位置对齐。

## 当前数据统计

当前生成结果记录在 `manifest.json` 和 `manifest.csv` 中：

```text
进入处理的 SVG：492 个
成功渲染并生成配对：475 个
渲染失败：17 个
clean PNG：475 张
每个 JPEG 质量目录：475 张
JPEG/clean pair 总数：1,900 对
```

渲染失败的 SVG 不会进入配对数据。失败原因记录在 `manifest.json` 的 `failures` 字段中。

## 目录结构

```text
jpeg_pairs_512/
├── clean_png/       # SVG 渲染得到的 512×512 RGB clean reference
├── jpeg_q10/        # JPEG quality=10 的 degraded 输入
├── jpeg_q20/        # JPEG quality=20 的 degraded 输入
├── jpeg_q30/        # JPEG quality=30 的 degraded 输入
├── jpeg_q50/        # JPEG quality=50 的 degraded 输入
├── manifest.csv     # 每个 SVG、PNG 和 JPEG 的对应关系
├── manifest.json    # 参数、哈希、失败列表和对应关系
├── README.txt       # 简要说明
└── DATA_HANDOFF_CN.md
```

同一个样本在不同目录中的文件名相同。例如：

```text
clean_png/example__HASH.png
jpeg_q30/example__HASH.jpg
```

因此可以通过相同的文件名配对 clean PNG 和 JPEG 输入。更完整的对应关系以 `manifest.csv` 为准。

## JPEG quality factor 的含义

当前生成了四种 JPEG quality：

```text
Q = 10, 20, 30, 50
```

Q 是 Pillow JPEG 编码器的质量参数，不是跨所有编码器完全统一的物理压缩率。通常 Q 越低，压缩越强，信息损失和伪影越明显：

- `Q=10`：强压缩，块效应、振铃和细节损失通常最明显；
- `Q=20`：较强压缩；
- `Q=30`：中等偏强压缩，适合观察明显伪影；
- `Q=50`：中等压缩，适合观察保边与轻中度伪影抑制。

当前 JPEG 使用：

```text
subsampling=2，即常见的 4:2:0 色度子采样
```

因此数据中同时包含亮度压缩和色度分辨率降低带来的影响。若后续需要单独研究色度子采样，可以另外生成 `subsampling=0` 的 4:4:4 数据，但不要与当前数据混淆。

## 推荐的第一轮实验

先使用 `jpeg_q30` 和 `jpeg_q50`：

```text
jpeg_q30/*.jpg → WLS(p0)
jpeg_q30/*.jpg → V8

jpeg_q50/*.jpg → WLS(p0)
jpeg_q50/*.jpg → V8
```

其中 `p0` 必须使用 V8 训练时对应的固定 WLS 参数。

比较两类结果：

1. **算子一致性**：V8 输出与相同参数 WLS 输出的差异；
2. **应用效果**：WLS/V8 输出与 `clean_png` 的差异。

这里的 clean PNG 是由 SVG 渲染得到的实验参考，不是 WLS 算子的 target。V8 当前仍然是原来训练得到的固定参数 WLS 模拟网络，没有使用这些 JPEG/clean pairs 重新训练。

## 重要限制

- `clean_png` 是 SVG 光栅化后的参考图，不是现实世界中某张 JPEG 的原始文件恢复结果。
- JPEG quality factor 依赖当前使用的 Pillow/libjpeg 编码器；更换编码器后，Q 值不能直接认为完全等价。
- 透明区域已经合成到白色背景；当前数据不用于研究 alpha 通道恢复。
- 17 个无法渲染的 SVG 没有进入数据集。
- 当前数据适合受控的合成 JPEG 实验。Office-Home 中已有的 JPEG 图片仍可作为真实来源的外部输入，但不能与这里的 clean PNG 混为同一类数据。

## 数据生成脚本

配对由上一级目录中的脚本生成：

```text
../prepare_clipart_jpeg_pairs.py
```

默认参数为：

```text
输入：openclipart_500/svg_raw
输出：openclipart_500/jpeg_pairs_512
尺寸：512×512
质量：10、20、30、50
子采样：4:2:0
背景：白色
```
