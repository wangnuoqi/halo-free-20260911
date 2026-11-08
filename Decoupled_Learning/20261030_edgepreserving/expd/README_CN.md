# Exp D: exp7C / NAFNet / VIVNet 推理耗时对照

`benchmark_inference.py` 使用同一批 20260818 test-200 输入，固定 batch=1、FP32、相同 GPU。输入在计时前读入并传至 GPU；每张图预热后用 CUDA events 只计前向，不包括 TIFF 解码、主机到显存拷贝、输出指标计算或文件保存。exp7C 分别记录完整前向、低分辨率网络加 guided filter（coarse）和三阶段 refine；后两项是诊断数据，完整前向才用于模型比较。结果写入 `results/inference_fp32.json`。

在 Windows CUDA 的 `yolo_clean` 环境执行（下面将物理 GPU 1 映射为进程内 `cuda:0`）：

```bash
wintrain '$env:CUDA_VISIBLE_DEVICES="1"; $env:PYTHONUNBUFFERED="1"; conda run --no-capture-output -n yolo_clean python -u D:\ResearchCode\halo-free\Decoupled_Learning\20261030_edgepreserving\expd\benchmark_inference.py --device cuda:0 --samples 5 --warmup 5 --repeats 20'
```

## 2026-11-03 实测

RTX 3090 物理 GPU 1，5 张 test 图均为 512×512；每张预热 5 次、重复 20 次。下表取各图前向耗时中位数，再取 5 张图的中位数。实际 checkpoint 均成功严格加载；三个模型均在 eval/inference mode，完整输出均截断到 `[0,1]`。

| 模型/阶段 | 参数量 | 前向耗时中位数 |
| --- | ---: | ---: |
| exp7C 完整流程 | 71,522 | 13.613 ms |
| exp7C coarse（诊断） | - | 2.966 ms |
| exp7C 三阶段 refine（诊断） | - | 10.708 ms |
| NAFNetLocal | 17,110,753 | 65.633 ms |
| VIVNet | 984,014 | 54.583 ms |

原始记录在 `results/inference_fp32.json`，包括每张图的 20 次计时。exp7C 完整前向约为 NAFNet 的 0.21 倍、VIVNet 的 0.25 倍。因此在这组共同推理条件下，没有复现“exp7C 比两者慢”的现象。此前看到的耗时若包含数据读取、GPU 传输、保存、训练前后向，或使用了不同精度、图像尺寸、设备，就不能直接对比。原始 exp7C 训练脚本里的 `t3-t1` 也包含 H2D 传输，不是纯前向计时。

exp7C 自身最大的计算段是三阶段串行全分辨率 refine；每阶段的 `compute_sobel_grad()` 还在前向中临时构造两个 Sobel kernel。这个结构仍有优化空间，但当前数据不支持把它列为相对 NAFNet/VIVNet 的速度故障。

此次测速时 Exp C 训练进程仍占用四张卡，GPU 争用使绝对耗时有波动；结果是并发条件下的对照。同协议前一次 5 图测试的中位数分别为 17.458、63.664、54.530 ms，速度排序一致。需要正式发布 latency 时，应在训练结束后用上面的同一命令重跑。比较中不使用 `DataParallel`，因为单图推理的额外分发会改变耗时。
