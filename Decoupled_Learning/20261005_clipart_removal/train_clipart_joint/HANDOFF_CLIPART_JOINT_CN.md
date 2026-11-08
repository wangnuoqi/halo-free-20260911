# Clip-art JPEG 联合训练 Handoff

## Scope

本目录用于在 OpenClipart 的 JPEG 配对数据上训练一个新的 clip-art 去伪影模型。训练结构遵守 `20260815_edgepreserving` 的 `exp7C_joint_fidelity_20260818.py`：ConvGuided 第一阶段加 GroupNorm 三阶段 Refine，使用 coarse fidelity loss，并从训练第一轮开始联合更新两个模块。

本实验的监督目标从原来的 WLS 输出改为 clean clip-art PNG 的亮度通道。它验证的是同一网络结构在 JPEG clip-art 复原任务上的可训练性，不代表网络继续模拟固定参数 WLS。

## Current Status

**Implemented, not run:** 训练脚本、固定数据划分、启动脚本和本 handoff 已创建；训练 checkpoint 尚未产生。训练依赖 `openclipart_5000/jpeg_pairs_512` 已经生成完成，并且其中存在可配对的 `clean_png` 与 `jpeg_q30` 文件。

下载目录中的非法 SVG 会在配对生成阶段被跳过，因此最终训练样本数以实际存在的 PNG/JPEG 同名配对为准。

## Verified Artifacts

网络和 loss 定义取自：

```text
/mnt/d/researchcode/halo-free/Decoupled_Learning/20260815_edgepreserving/experiments/exp7C_joint_fidelity_20260818.py
```

clip-art 训练代码：

```text
train_clipart_joint.py
```

启动脚本：

```text
run_train_clipart_q30.sh
```

固定数据划分：

```text
../openclipart_5000/train_list.txt
../openclipart_5000/val_list.txt
../openclipart_5000/test_list.txt
../openclipart_5000/split_info.json
```

三个 list 每行保存一个不带扩展名的 pair stem。当前划分由 4653 个完整 pair 生成：3722 train、465 validation、466 test，随机种子为 `20261008`。训练脚本直接读取这些文件，不会在启动时重新随机划分。

初始化 ConvGuided checkpoint：

```text
../20260815_edgepreserving/models/convguided_20260818_l0p500_a1p400/convguided_filter_best_20260818_l0p500_a1p400.pth
```

代码已经通过：

```bash
python3 -m py_compile train_clipart_joint.py
bash -n run_train_clipart_q30.sh
```

这只是语法验证，不是训练完成证明。

## Reproduction

先生成配对数据：

```bash
cd /mnt/d/researchcode/halo-free/Decoupled_Learning/20261005_clipart_removal
bash prepare_openclipart_5000_pairs.sh
```

然后训练 Q30：

```bash
cd /mnt/d/researchcode/halo-free/Decoupled_Learning/20261005_clipart_removal
bash train_clipart_joint/run_train_clipart_q30.sh
```

等价的 Python 命令：

```bash
/home/nuoqi/miniconda3/envs/halofree/bin/python -u \
  train_clipart_joint/train_clipart_joint.py \
  --data-root openclipart_5000/jpeg_pairs_512 \
  --quality 30 \
  --epochs 80 \
  --batch-size 1 \
  --device cuda:0 \
  --output-dir train_clipart_joint/checkpoints/q30
```

当前固定划分是按同名源图一次性随机划分：3722 train、465 validation、466 test，随机种子为 `20261008`。同一张图的 Q10/Q20/Q30/Q50 版本都由同一份 list 决定集合归属，因此不会让同一内容跨集合。当前默认先训练 Q30；Q10、Q20、Q50 应作为独立实验运行并写入不同的 `--output-dir`。如需显式指定 list 目录，可传 `--list-dir /mnt/d/researchcode/halo-free/Decoupled_Learning/20261005_clipart_removal/openclipart_5000`。

## Design and Decision Context

每个样本的数据流是：

```text
jpeg_q30/example.jpg
       ├── Y 通道，512×512，作为 x_hr
       └── Y 通道 Lanczos 缩小到 256×256，作为 x_lr

clean_png/example.png
       └── Y 通道，作为 clean GT
```

网络只处理 Y 通道，以保持原始模型的单通道结构。JPEG 的 RGB 文件仍然作为输入来源，训练时通过 YCbCr 转换取亮度；这与当前 `DeepGuidedFilter`、`ConvGuidedFilter` 和 Sobel edge loss 的单通道假设一致。

保留的原始训练逻辑包括：

1. 从已经训练好的 ConvGuided checkpoint 初始化第一阶段。
2. 新建 `CascadeRefine3Stage`，Refine 从头初始化。
3. ConvGuided 参数设置为可训练，学习率 `3e-4`。
4. Refine 参数学习率为 `3e-3`。
5. 两个模块从第一个 epoch 开始使用同一个 optimizer 联合更新。
6. 使用三个阶段的 edge/smooth/gradient loss，权重为 `0.1/0.3/0.6`。
7. 增加 `0.5 * SmoothL1(coarse, GT)`，保持 coarse output 的 fidelity。
8. 每 30 个 epoch 将学习率乘以 `0.5`，默认训练 80 个 epoch。
9. 根据 validation MSE 保存成对 checkpoint：

```text
checkpoints/q30/rf_3stage_best.pth
checkpoints/q30/convguided_joint_best.pth
```

最终 test 指标写入：

```text
checkpoints/q30/test_metrics.txt
```

## Pitfalls and Constraints

- `jpeg_qXX` 是 degraded input，`clean_png` 才是 GT；不能把 OpenClipart 原始 SVG 或已有 JPEG 当作 clean GT。
- 当前脚本做的是 JPEG 伪影复原，不是 RGB 三通道复原。颜色 Cb/Cr 没有进入网络。
- `x_lr` 必须从 degraded JPEG 的 Y 通道生成，不能从 clean PNG 生成，否则训练时会泄漏 clean 信息。
- 4:2:0 数据目录 `jpeg_pairs_512` 是当前主实验；4:4:4 数据应使用另一套 `--data-root` 和单独 checkpoint 目录。
- 训练前必须确认 `clean_png` 和对应 `jpeg_qXX` 的 stem 完全匹配。脚本只使用匹配成功的文件。
- 不要把不同 quality 的同一源图拆到不同数据集合中。若要混合 Q10/Q20/Q30/Q50，应先按源图划分，再在每个集合内混合质量版本。
- 训练结束后推理必须同时加载 `convguided_joint_best.pth` 和 `rf_3stage_best.pth`，不能只加载 Refine。
- 该目录尚无训练结果；任何 PSNR、MSE 或视觉结论都必须等实际训练和独立测试完成后再报告。

## Open Questions

- Q30 训练是否比直接使用原来的 WLS imitation checkpoint 更能接近 clean PNG？
- Q10、Q20、Q30、Q50 应分别训练，还是在无泄漏划分下混合质量训练？
- 只训练 Y 通道时，彩色边界残留的误差有多少来自输入 Cb/Cr？
- 3-stage Refine 是否比只使用 ConvGuided coarse output 更能减少 JPEG 伪影，同时保留细线？

## Next Actions

1. 完成 `openclipart_5000/jpeg_pairs_512` 的配对生成并检查有效 pair 数量。
2. 保留 `openclipart_5000/*_list.txt` 作为所有 quality 实验的固定划分来源。
3. 先运行 `run_train_clipart_q30.sh`，确认 GPU、checkpoint 和数据读取无误。
4. 检查 `train_clipart_joint/checkpoints/q30/` 是否出现两个最佳 checkpoint 和 `test_metrics.txt`。
5. 使用独立推理脚本同时加载两个 checkpoint，输出 input、clean GT 和 model output 的彩色可视化；Y 输出重新与输入 JPEG 的 Cb/Cr 合成，仅用于展示，评价仍单独报告 Y 指标。
6. 在 Q10、Q20、Q50 上重复实验，并保持同一组 list 划分。
