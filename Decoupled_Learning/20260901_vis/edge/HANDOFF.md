# Edge 保边模型测试与可视化 Handoff

## 范围

本目录是独立的保边模型推理、结果整理和可视化工作区，不负责重新训练模型。当前比较对象为：

- `baseline_original`：原始 ConvGuided + 原始三阶段 RF。
- `exp7c_recipe_*`：Exp7C 的 gated 三阶段 RF/联合 ConvGuided 模型，在对应 WLS 标签集上训练的版本。

## 目录结构

```text
edge/
├── evaluation/                 Python 推理、下采样和回归测试
├── models/                     网络定义与 checkpoint
├── visualization/              MATLAB 彩色重建、三联图、误差热力图
├── outputs/                    推理输出和 metrics.txt
├── README.md                   快速说明
└── HANDOFF.md                  本文档
```

## Evaluation

### `evaluation/run_inference_compare.py`

运行 Baseline 和 Exp7C 两套完整 pipeline。脚本解析 `testinput`、`testlabel` 和 LR 目录，按样本名匹配文件；缺少 LR 时在线生成半分辨率输入；加载对应协议的 ConvGuided/RF checkpoint；输出三组 float32 TIFF：`baseline_original/`、`exp7c_recipe/`、`gt/`，并写入 `metrics.txt`。

指标包括 mean MSE、由 mean MSE 换算的 PSNR，以及逐图 PSNR 的算术平均。`test_vivo` 根目录包含多个参数组，必须指定 `--parameter-tag`。

示例：

```bash
cd /mnt/d/nuoqi/halo-free/Decoupled_Learning/20260901_vis/edge
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n halofree \
  python evaluation/run_inference_compare.py \
  --data-root /mnt/d/nuoqi/S2R-HDR/data/test_vivo \
  --parameter-tag l1p000-a1p600 \
  --output-root outputs/test_vivo/wls_l1p000_a1p600 \
  --device cuda:0
```

### `evaluation/prepare_test_vivo_downsample.py`

调用 `total-pipeline-v1/downsample_hdr.py`，为 `test_vivo` 的 `l0p250-a1p200`、`l0p500-a1p400`、`l1p000-a1p600` 三个输入组生成对应的 `_downsample` LR 目录。

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n halofree \
  python evaluation/prepare_test_vivo_downsample.py
```

### `evaluation/prepare_test_fei_downsample.py`

同样调用 `downsample_hdr.py`，处理 `test_fei/generated_20260813`、`generated_20260818` 和扁平 `testinput` 协议。只生成 LR，不运行模型。

### `evaluation/test_run_inference_compare.py`

Pytest 回归测试，检查网络定义导入、数据目录解析、样本名匹配、模型协议映射，以及 `test_fei`/`test_vivo` 参数目录兼容性。

```bash
conda run --no-capture-output -n halofree pytest -q evaluation/test_run_inference_compare.py
```

## Models

| 目录 | 数据协议 | ConvGuided | RF |
|---|---|---|---|
| `models/baseline_original/` | 原始 mixed label | `convguided_filter_best_mix.pth` | `rf_3stage_best_otsu_v3.pth` |
| `models/baseline_20260813/` | WLS `lambda=1.0, alpha=1.6` | `convguided.pth` | `rf.pth` |
| `models/baseline_20260818/` | WLS `lambda=0.5, alpha=1.4` | `convguided.pth` | `rf.pth` |
| `models/exp7c_recipe_mixed/` | mixed label 的 Exp7C/联合模型 | `convguided_joint_best_20260818_l0p500_a1p400.pth` | `rf_3stage_best_20260818_l0p500_a1p400.pth` |
| `models/exp7c_recipe_20260813/` | WLS `1.0, 1.6` | `convguided.pth` | `rf.pth` |
| `models/exp7c_recipe_20260818/` | WLS `0.5, 1.4` | `convguided.pth` | `rf.pth` |

Definition 文件含义：

- `baseline_original/convguided_definition.py`：原始 ConvGuided 网络定义。
- 各 `baseline_*/refine_definition.py`：原始三阶段 RF 网络定义副本。
- 各 `exp7c_recipe_*/exp7c_recipe_definition.py`：Exp7C 的 gated 三阶段 RF 与联合 ConvGuided 定义，包含 Sobel 梯度、delta/gate 和逐像素参数更新。

权重必须和同目录 definition 配套加载，不能只根据文件名替换 checkpoint。

## Outputs

推理结果按“测试数据/参数”组织：

```text
outputs/test_vivo/wls_l0p250_a1p200/
outputs/test_vivo/wls_l0p500_a1p400/
outputs/test_vivo/wls_l1p000_a1p600/
outputs/test_fei/wls_l0p250_a1p200/
outputs/test_fei/wls_l0p500_a1p400/
outputs/test_fei/wls_l1p000_a1p600/
```

每个参数目录包含：

```text
baseline_original/*.tiff
exp7c_recipe/*.tiff
gt/*.tiff
metrics.txt
```

已记录的逐图 PSNR 平均值：

| 测试集/参数 | Baseline | Exp7C |
|---|---:|---:|
| test_vivo / 0.25, 1.2 | 50.6661 dB | 53.7203 dB |
| test_vivo / 0.5, 1.4 | 42.1052 dB | 45.2534 dB |
| test_vivo / 1.0, 1.6 | 39.8487 dB | 41.7569 dB |
| test_fei / 0.25, 1.2 | 51.9895 dB | 55.3598 dB |
| test_fei / 0.5, 1.4 | 43.8644 dB | 48.2153 dB |
| test_fei / 1.0, 1.6 | 42.0038 dB | 45.1606 dB |

数值来源是各参数目录下的 `metrics.txt`。报告时要区分 `PSNR from mean MSE` 与逐图 PSNR 平均值。

## Visualization

### `visualization/test_vivo_modelvis_multi_dirs.m`

读取 `outputs/test_vivo/wls_*` 下的灰度 TIFF，并匹配 `S2R-HDR/data/test_vivo/hdrimg` 原始 HDR。使用设备归一化显示规则：

```text
comp = 9937 / (log(max(HDR)) * 1024)
```

为三个参数目录分别保存 Baseline、Exp7C、GT 的单图和三联图。`display_scale` 只改变可视化亮度，不改变推理输出和指标。

### `visualization/reconstruct_color_compare.m`

单一协议的彩色重建脚本。读取 HDR、模型输出和标签，执行 WLS/detail 重建、归一化和 sRGB gamma 校正，保存 `Baseline | Exp7C | GT` 三联图。

### `visualization/test_vivo_error_heatmaps.m`

计算并绘制 Baseline/Exp7C 相对于 GT 的误差：

```matlab
baseErr = mean(abs(base - gt), 3);
expErr  = mean(abs(exp7 - gt), 3);
improvement = baseErr - expErr;
```

每个样本单独计算一个共享范围：

```matlab
errMax = max([baseErr(:); expErr(:)]);
```

同一张三联图内 Baseline 和 Exp7C 共用 `[0, errMax]`，不同样本允许有不同范围。GT 不参与最大值计算，因为 `GT-GT=0`。每张图旁的 `_error_range.txt` 保存实际范围。

### `visualization/test_fei_error_heatmaps.m`

这是 `test_vivo_error_heatmaps.m` 的 `test_fei` 入口包装器。它读取 `visualization/test_fei/wls_*` 下已有的 Baseline、Exp7C 和 GT 彩色图，并在每个参数目录下生成 `heatmaps/`。

运行：

```matlab
cd('D:\nuoqi\halo-free\Decoupled_Learning\20260901_vis\edge\visualization');
test_fei_error_heatmaps;
```

改进图定义为 `baseline_error - exp7c_error`：蓝色表示 Baseline 误差更小、Baseline 更好；红色表示 Exp7C 误差更小、Exp7C 更好；白色表示两者接近。

### 参考文件

- `test_tmcode_modelvis_multi_dirs_reference.m`：旧版多模型目录可视化参考。
- `test_tmcode_modelvis_params_reference.m`：旧版按 WLS 参数解析的参考。
- `wls.m`：MATLAB WLS 保边滤波核心函数。
- `test_fei.zip`、`test_vivo.zip`：历史脚本/结果归档，不是当前入口。

## 复现顺序

1. 确认输入、标签和 LR 目录存在。
2. 必要时运行对应的 `prepare_test_*_downsample.py`。
3. 运行 `run_inference_compare.py` 生成 float32 TIFF 和指标。
4. 在 MATLAB 中运行对应彩色重建脚本。
5. 运行 `test_vivo_error_heatmaps.m` 生成误差热力图。

## 注意事项

- MATLAB 脚本使用 `D:\nuoqi\...` Windows 路径；Python 使用 `/mnt/d/nuoqi/...` WSL 路径。
- 模型、definition、标签参数必须匹配；不要用 mixed label 权重直接替代 20260813/20260818 权重。
- 彩色 JPEG 仅用于观察；严格误差分析应使用 evaluator 输出的 float32 TIFF。
- 当前环境没有 MATLAB，MATLAB 脚本只完成静态检查，实际渲染需在 MATLAB 中执行。

## 当前状态

- Python evaluator、参数解析、样本匹配和回归测试已整理。
- `test_vivo`、`test_fei` 三组参数的推理输出和指标文件已存在。
- 热力图已按“单张三联图内部共享归一化范围”实现。
