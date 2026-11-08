# Exp D: OpenClipart 推理耗时核对

比较项目内已训练的 `exp3-b`（exp7C 风格 ConvGuided + 三阶段 Refine）与 VIVNet，使用同一批 Q30 test JPEG。脚本分别记录预加载 GPU 输入后的 FP32 前向，以及 `evaluate_all_models.py` 中现有 `predict_*` 整条路径的同步耗时。后者包含 JPEG 解码、预处理、CPU/GPU 传输与输出回传。每项都是单 GPU、batch=1，避免多卡分发影响单图延迟。

```bash
wintrain '$env:CUDA_VISIBLE_DEVICES="1"; $env:PYTHONUNBUFFERED="1"; conda run --no-capture-output -n yolo_clean python -u D:\ResearchCode\halo-free\Decoupled_Learning\20261005_clipart_removal\expd\benchmark_clipart_inference.py --device cuda:0 --samples 5 --warmup 3 --repeats 10'
```

原评估记录 `baselines/predictions/q30p160/summary.csv` 的 `exp3-b` 约 0.633 s/张、VIVNet 约 0.063 s/张，但该次使用 batch=4 和四卡 `DataParallel`。另一次 `evaluation/q30_non_swin_timed/summary.csv` 的 `exp3-b` 约 0.035 s/张，协议不同。不能直接将这些数字解释成模型前向的速度差。

## 当前项目的 clip-art 权重

2026-11-03 在物理 GPU 1（RTX 3090）上测 5 张 test JPEG，单卡 batch=1、FP32，每张预热 3 次、重复 10 次。表中为先取逐图中位数、再取 5 张图中位数。

| 模型 | 参数量 | 纯 GPU 前向 | 原 `predict_*` 整条路径 |
| --- | ---: | ---: | ---: |
| `exp3-b`（exp7C 风格，`exp3-b_joint/joint_best.pth`） | 75,910 | 15.640 ms | 34.616 ms |
| clip-art VIVNet（`net_g_latest.pth`） | 985,166 | 54.039 ms | 63.512 ms |

`exp3-b` 的 base 为 4.017 ms、三阶段 refine 为 12.325 ms。完整逐次计时、checkpoint 绝对路径和实验条件在 `results/clipart_fp32.json`。原 `predict_ours()` 对同一 JPEG 先读 YCbCr、再读 RGB；RGB 路径只用后者。已移除这次多余解码。修正前同协议计时在 `results/predecode_fix.json`：整条路径 38.307 ms/张，修正后 34.616 ms/张。由于训练并发，不能把两次运行差值全部视为精确收益。当前路径仍在 CPU 上用 Lanczos 生成 LR，再将 HR/LR 传至 GPU，最后回传输出；这解释了前向以外的耗时。

旧的 `0.633 s/张` 记录使用四卡 DataParallel，base 和 refine 各触发一次多卡分发。用原 `evaluate_all_models.py` 入口，在同一批前 20 张 test 图、同为 batch=4 的条件下复测：

| 方法 | 单卡 inference/张 | 四卡 DataParallel inference/张 |
| --- | ---: | ---: |
| `exp3-b` | 0.0750 s | 0.6397 s |
| clip-art VIVNet | 0.1486 s | 0.1503 s |

两次 `exp3-b` 的 MSE 均为 `0.0002964166362062315`，确认权重与输出一致。四卡复测几乎重现旧 `0.633 s/张`，单卡则快约 8.5 倍。可见异常主要由该模型当前的多卡推理路径触发；base/refine 分别包裹 DataParallel 是可见的额外分发，但这次测量尚未把 0.64 s 逐算子归因。原入口复测文件分别在 `results/dp_probe/summary.csv` 与 `results/single_probe/summary.csv`。这两次测速时 Exp C 训练仍占用 GPU，正式速度报告应在空闲设备上复测。

### DataParallel 内部定位

后续在同一批四张图、batch=4 的诊断显示：单卡完整前向 28.3 ms/批，而原样双模块 DataParallel 为 2446 ms/批。分开测时，base 和 refine 的 DataParallel 调用各约 1.2 s/批。让 refine 只返回 `final` 后仍约 1.24 s/批；将 base+refine 合并为一次 DataParallel 后仍约 2.32 s/批，因此多余的中间图 gather 和两次分发都不是主因。见 `results/dataparallel_diagnosis.json`。

对 DataParallel 的内部方法同步计时，base 的 `scatter/replicate/parallel_apply/gather` 分别为 2.3/8.0/1190.3/5.2 ms；refine 分别为 4.1/5.4/1214.5/31.2 ms。绝大部分延迟在 `parallel_apply`，见 `results/dp_phases.json`。四张卡逐张顺序运行同一模型，完整前向各为 16–17 ms，见 `results/direct_gpus.json`，排除了某一张卡天然慢的解释。

最后保留相同的四卡 `scatter/replicate/gather`，仅把 replica 的多线程 `parallel_apply` 改成顺序执行：原线程版本 2468 ms/批，顺序版本 133 ms/批，输出最大差异为 0。见 `results/parallel_apply_modes.json`。因此已定位到当前 Windows/PyTorch 环境下这套模型的 DataParallel 多线程执行路径；目前没有证据进一步区分 Python 线程调度、CUDA 驱动调用与大量小算子之间的具体占比。实用推理应使用单卡原模型；这个自定义顺序 DataParallel 只是诊断对照，不是部署方案。

## WLS 训练权重推 clip-art

此目录没有找到 clip-art 训练的 NAFNet checkpoint。下面另测了 **WLS 训练**的 exp7C、NAFNet、VIVNet 已有 checkpoint，把相同的 Q30 JPEG 转成 Y 通道后送入三个单通道模型。此表只比较运行速度，不代表 clip-art 去伪影质量。

| 模型 | 参数量 | 纯 GPU 前向 |
| --- | ---: | ---: |
| WLS exp7C | 71,522 | 13.590 ms |
| WLS NAFNetLocal | 17,110,753 | 64.760 ms |
| WLS VIVNet | 984,014 | 53.896 ms |

逐次记录在 `results/wls_models_on_clipart_fp32.json`。复现命令：

```bash
wintrain '$env:CUDA_VISIBLE_DEVICES="1"; $env:PYTHONUNBUFFERED="1"; conda run --no-capture-output -n yolo_clean python -u D:\ResearchCode\halo-free\Decoupled_Learning\20261005_clipart_removal\expd\benchmark_wls_models_on_clipart.py --device cuda:0 --samples 5 --warmup 3 --repeats 10'
```

`infer_exp7c_clipart/infer_exp7c_clipart.py` 虽然文件名含 exp7C，当前实际加载的是 exp8B mixed-label checkpoint；不能把它的输出或耗时直接标成 exp7C。

## 三个任务的 4K 单图耗时与参数量

`benchmark_4k_tasks.py` 使用三个项目各自的真实测试输入，放大到 3840×2160 后进行单 GPU、batch=1、FP32 推理。项目当前没有原生 4K 测试图，因此这是分辨率压力测试，图像会被拉伸到 16:9；不用于比较图像质量。所有模型统一切成 1024×1024 tile、128px 重叠，计时包括模型前向与 tile 拼接，不包括读图、缩放、H2D/D2H 和 checkpoint 加载。每个模型预热 1 次、重复 3 次，报告中位数。

2026-11-04 在 RTX 3090、GPU 2 上测量。Exp C 的四卡训练同时运行，因此这些是训练负载下的观测值，空闲卡上的延迟会不同。参数量是对应完整推理模型（Exp7C/exp3-b/Stage3 将两个子网络相加）的可训练参数总数。

| Task | Model | Parameters (M) | 4K Inference Time (s/image) |
| --- | --- | ---: | ---: |
| Edge Preserving | Ours | 0.071522 | 0.846 |
| Edge Preserving | NAFNet | 17.110753 | 2.837 |
| Edge Preserving | VIVNet | 0.984014 | 4.329 |
| Clip-Art Removal | exp3-b | 0.075910 | 0.931 |
| Clip-Art Removal | VIVNet | 0.985166 | 4.444 |
| Texture Removal | Three-Stage Joint | 0.075610 | 0.527 |
| Texture Removal | RTV (Official) | 1.394944 | 0.857 |
| Texture Removal | NAFNet | 17.111907 | 2.970 |

相同口径下，Exp7C/exp3-b 在这三项已有对照中都比 VIVNet/NAFNet 快；它们约 7.6 万参数，VIVNet 约 98.5 万，NAFNet 约 1,711 万。旧结果中 Exp7C/exp3-b 显得更慢，原因是四卡 `DataParallel` 在当前 Windows/PyTorch 环境的 `parallel_apply` 多线程路径异常昂贵，而不是参数较少却计算天然更慢：已有分阶段诊断中，单卡前向约 16–28 ms，DP 的 base/refine `parallel_apply` 则各约 1.2 s。4K 全图初测也观察到 Exp7C 约 0.45 s、NAFNet 约 1.46 s；VIVNet 全图测试显存达到 24 GB 上限附近，未完成，因此正式对照统一改用分块。

完整的逐次耗时、checkpoint、GPU 和协议记录位于 `results/4k_task_benchmarks.json`。复现命令：

```bash
wintrain '$env:CUDA_VISIBLE_DEVICES="2"; $env:PYTHONUNBUFFERED="1"; conda run --no-capture-output -n yolo_clean python -u D:\ResearchCode\halo-free\Decoupled_Learning\20261005_clipart_removal\expd\benchmark_4k_tasks.py --task all --device cuda:0 --height 2160 --width 3840 --warmup 1 --repeats 3 --tile-size 1024 --tile-overlap 128'
```
