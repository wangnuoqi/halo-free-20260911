# Exp E：VIV-LR 引导的多窗口仿射重建

实验设计、结果和协议对照见 [HANDOFF_CN.md](HANDOFF_CN.md) 与 [EXPERIMENT_DETAILS_CN.md](EXPERIMENT_DETAILS_CN.md)。E1 已完成60K及完整checkpoint：实际目录为 `expe/runs/scratch/viv_gf_e1/seed42_ddpfix/`；最佳权重在55000步，验证PSNR=44.9715 dB，最终60000步为44.9624 dB。E0（原 LRNet + GF）现已加入统一训练入口，与 E1 做同协议对照；此前只运行1步并已暂停，没有可用 checkpoint，重跑需使用新输出目录。

## 模型与变体

E0 使用旧 Exp7C 的 24 通道四个 dilation LRNet（dilation 2/4/8/16）和原 ConvGuided；E1 及 A/B/C 变体使用 VIVNet 的宽度 32、三个尺度各 1 个 block 的 U-Net 主干预测 `y_lr`，不使用末端 4 个 refinement block。两者都接收原图的半分辨率配对 LR 图，HR guide 仍是原图。E0/E1 的 Base 都使用当前统一训练协议；新 A 系列五个变体从随机初始化联合训练 LRNet 与仿射估计模块。新增的外接三阶段 Refine/Joint 见 [TRAIN_REFINE_JOINT_CN.md](TRAIN_REFINE_JOINT_CN.md)。

| 变体 | 改动 |
| --- | --- |
| `viv_gf_e1` | VIV-LR + 原 3x3 可学习 box/1x1 `a`/双线性 `a,b` |
| `viv_gf_a1_box3`, `viv_gf_a1_box9` | 固定、归一化的单窗口对照 |
| `viv_gf_a1_fixed_boxes` | 固定均值 3x3/9x9 双窗口结构；归档 global32 结果已完成，本轮按 global64 重训 |
| `viv_gf_a1_stats`, `viv_gf_a2_stats_edge` | E1 完整可学习 3x3 统计分支加独立 9x9 分支；A2 只在门控增加四张边缘图 |
| `viv_gf_a1_direct`, `viv_gf_a2_direct_edge` | 可学习 3x3/9x9 分支直接预测 `a,b`；A2 只在门控增加相同边缘图 |
| `viv_gf_b` | 旧固定统计分支 + HR 像素对邻近 LR 仿射模型的可学习选择；尚未迁移到新 A2 |
| `viv_gf_c1`, `viv_gf_c2`, `viv_gf_c3` | B + 一、二、三轮 HR `a,b` 联合重估计 |
| `viv_gf_c2_aonly` | C2 中只允许 `a` 更新的对照 |

`model(guide_hr, input_lr)` 返回 HR coarse；`return_aux=True` 额外返回 `y_lr`、候选系数、窗口权重、HR 融合权重和阶段输出。推理输入中的 Otsu 边缘只由 HR 原图和预测 `y_lr` 生成，不使用 GT。

已完成的旧固定统计 A1/A2 为双卡每卡16、global32，不作为 E1 的同协议单因素结果；代码和 checkpoint 已移到[旧 A 归档](archive/legacy_a/README_CN.md)。本轮串行训练包括统计版 A1 → A2、直接预测版 A1 → A2，以及固定双窗口 A1，共五个变体。第五个在新目录按 global64 从头训练，不从旧 global32 checkpoint 续训；固定 box3/box9 单窗口另作消融。训练入口与对照边界见 [可学习多卷积核实验说明](LEARNED_KERNEL_EXPERIMENTS_CN.md)。

## 训练协议

复用 `../expc/src/train_expc.py` 的 20260818 预计算配对 2x LR TIFF、HR input、WLS target、固定 train 内 200 张**全图**验证、128 HR crop、seed 42、global batch 64、AdamW、BF16、60K cosine 调度及 best/latest/final checkpoint。E1 是四卡 DDP/NCCL、每卡16；A 系列使用双卡 DDP/NCCL、每卡 micro-batch16、梯度累积2，即有效 global batch64。两者每 rank 8 workers，A 脚本核对 E1 实际协议。`pin_memory` 默认关闭。默认直接训练到 60K，每 5K 验证一次；独立 test-200 不参与选模型。每个变体的 `train_loss.csv` 记录首次、每 100 step 及结束时的训练 `loss/pixel/gradient/lr`，`history.csv` 记录全图验证指标。

下段初始化与冻结说明适用于原有 paired 方案；新 A 系列五个变体均不加载 E1 权重，也不冻结 LRNet，使用上述独立脚本从头训练。

E1 先训练，用 `best.pth` 初始化 A/B/C 的 LRNet。后续变体加 `--init-lr-checkpoint` 和 `--freeze-lr`，只训练系数模块；模型的 frozen LRNet 在训练模式下保持 eval。结构比较统一用 `--loss-mode base`：沿用 20260815 原 coarse 的 MSE + 反射填充 Sobel 向量梯度 L1 动态平衡，公式为 `MSE + Sobel梯度L1 / max(detach(Sobel梯度L1 / (MSE + 1e-8)), 0.1)`。原代码下限为 `1e-8`；这里将下限提高到 `0.1`，避免近零梯度误差的浮点残差被放大。正常比例大于 `0.1` 时公式与原代码相同。`train_loss.csv` 的 `pixel` 是 MSE，`gradient` 是缩放前的 Sobel 误差，`loss` 是总项。选定结构后再比较 `edge`、`staged`、`staged_regularized`；这些模式保留独立的 Charbonnier/有限差分目标，不是仅在新版 `base` 上加一个项。joint 微调从冻结消融选出的**同变体完整 checkpoint**开始，用 `--init-model-checkpoint` 加载到新输出目录并解冻 LRNet；它是新优化运行，不继承旧 optimizer/scheduler。

## 每个实验的重训入口

所有最终独立测试报告统一使用每张图中心 **512×512 HR crop**，配对输入为 **256×256 LR**。模型选择仍只使用训练集内部 validation；test-200 只用于最终汇报。E1 三阶段链与历史原 Exp7C 已在相同 test-200 上的 512 crop 对照见 [test-200 报告](evaluation/test_200_512/README_CN.md)，训练协议差异也列于该报告。

在 `20261030_edgepreserving` 目录执行以下命令。`wintrain_command.py` 用于原有 E/A/B/C 变体；2×2 中除 E1 新配方外的三格改用 `matched_2x2_wintrain.py` 生成 Windows 双卡命令。E1 的已完成 Base checkpoint 在 `runs/scratch/viv_gf_e1/seed42_ddpfix/best.pth`。E0 新配方必须从随机初始化按双卡每卡16、梯度累积2、60K更新重训，独立输出到 `seed42_exposure3840000`；旧 `seed42_rerun` 权重不能当作对齐结果。旧配方 E0/E1 使用同样有效 batch64和60K更新，输出到 `legacy_matched_exposure3840000_v5`。其余系数变体用 `paired` 加载并冻结同一个 E1 LRNet，系数模块随机初始化；需要时也可用 `scratch` 从随机初始化重训变体。

| 实验 | `--variant` | 增加/删除的能力 |
| --- | --- | --- |
| E0 | `exp7c_e0` | 旧24通道 dilation LRNet + 原 GF，同协议架构对照 |
| E1 | `viv_gf_e1` | VIV-LR + 原 GF |
| A1 box3 | `viv_gf_a1_box3` | 固定 3x3 box |
| A1 box9 | `viv_gf_a1_box9` | 固定 9x9 box |
| A1 fixed boxes | `viv_gf_a1_fixed_boxes` | 固定均值 3x3/9x9 双窗口；旧 global32 checkpoint 在归档，本轮匹配重训 |
| A1 stats | `viv_gf_a1_stats` | E1 统计分支 + 可学习 9x9 分支及门控 |
| A2 stats edge | `viv_gf_a2_stats_edge` | 统计版 A1 门控加入梯度/Otsu |
| A1 direct | `viv_gf_a1_direct` | 双窗口直接预测 `a,b` |
| A2 direct edge | `viv_gf_a2_direct_edge` | 直接预测版 A1 门控加入梯度/Otsu |
| B | `viv_gf_b` | HR 邻域系数选择 |
| C1 | `viv_gf_c1` | 一轮 HR `a,b` 重估计 |
| C2 a-only | `viv_gf_c2_aonly` | 两轮、只更新 `a` |
| C2 | `viv_gf_c2` | 两轮 HR `a,b` 重估计 |
| C3 | `viv_gf_c3` | 三轮 HR `a,b` 重估计 |

当前 E1 的 WSL 四卡 DDP/NCCL 正式训练命令如下；`--num-workers 8` 是每 rank 8 个，总共 32 个，未开启 `pin_memory`。从本目录执行，一次训练到 60K：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 TORCH_DDP_BACKEND=nccl PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_MODULE_LOADING=LAZY \
NCCL_SHM_DISABLE=0 OMP_NUM_THREADS=1 \
/home/nuoqi/miniconda3/envs/halofree/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 expc/src/train_expc.py \
  --parallelism ddp --variant viv_gf_e1 \
  --data-root /mnt/d/ResearchCode/S2R-HDR/data/20260818 \
  --cache-path expc/runs/0818_memmap_4gpu/train_0818_float32.npy \
  --output-dir expe/runs/scratch/viv_gf_e1/seed42 \
  --crop-size 128 --batch-size-per-gpu 16 --total-iter 60000 \
  --val-count 200 --val-every 5000 --save-every 100 \
  --num-workers 8 --precision bf16 --lr 3e-4 --min-lr 1e-6 \
  --seed 42 --loss-mode base
```

旧的手写 E1 四卡 5K 命令（保留供核对参数；其输出目录与下方重训入口不同）：

```bash
wintrain '$env:CUDA_VISIBLE_DEVICES="0,1,2,3"; $env:PYTHONUNBUFFERED="1"; $env:OMP_NUM_THREADS="1"; conda run --no-capture-output -n yolo_clean python -u D:\ResearchCode\halo-free\Decoupled_Learning\20261030_edgepreserving\expc\src\train_expc.py --parallelism dp --variant viv_gf_e1 --data-root D:\ResearchCode\S2R-HDR\data\20260818 --output-dir D:\ResearchCode\halo-free\Decoupled_Learning\20261030_edgepreserving\expe\runs\viv_gf_e1 --cache-path D:\ResearchCode\halo-free\Decoupled_Learning\20261030_edgepreserving\expc\runs\0818_memmap_4gpu\train_0818_float32.npy --crop-size 128 --batch-size-per-gpu 16 --total-iter 60000 --stop-after 5000 --val-count 200 --val-every 5000 --save-every 100 --num-workers 4 --precision bf16 --lr 3e-4 --min-lr 1e-6'
```

新 A 系列五格使用[独立双卡 DDP 脚本](LEARNED_KERNEL_EXPERIMENTS_CN.md)，从头训练并按有效 batch、总步数和样本呈现量对齐 E1。下面的旧 `paired` 入口仍用于固定 box 单窗口及 B/C 方案；只查看完整 `wintrain` 命令时，运行 `python expe/wintrain_command.py --phase paired --variant viv_gf_c2`。

```bash
eval "$(python expe/wintrain_command.py --phase scratch --variant viv_gf_e1)"
for variant in viv_gf_a1_box3 viv_gf_a1_box9 viv_gf_b viv_gf_c1 viv_gf_c2_aonly viv_gf_c2 viv_gf_c3; do
    eval "$(python expe/wintrain_command.py --phase paired --variant "$variant")" || break
done
```

重训单个变体的随机初始化对照、从显式短跑继续、指定独立随机种子，以及同变体 joint 微调：

```bash
eval "$(python expe/wintrain_command.py --phase scratch --variant viv_gf_c2 --run-id seed43 --seed 43)"
eval "$(python expe/wintrain_command.py --phase paired --variant viv_gf_c2 --resume --stop-after 60000)"
eval "$(python expe/wintrain_command.py --phase paired --variant viv_gf_c2 --run-id seed43 --seed 43 --e1-checkpoint 'D:\ResearchCode\halo-free\Decoupled_Learning\20261030_edgepreserving\expe\runs\scratch\viv_gf_e1\seed42\best.pth')"
eval "$(python expe/wintrain_command.py --phase joint --variant viv_gf_c2)"
```

配对结构对照统一使用 `base` loss。要做 loss 消融，把 `--loss-mode edge`、`staged` 或 `staged_regularized` 加到 `--phase paired --variant viv_gf_c2` 命令；输出在 `runs/paired/viv_gf_c2/<loss_mode>/seed42`，互不覆盖。`staged` 系列只对 C 的多阶段输出增加监督。`--phase joint` 默认从同变体 `paired/base/seed42/best.pth` 初始化，非默认来源请显式传 `--source-checkpoint`。每个阶段默认直接跑满 60K；只有显式指定较小的 `--stop-after` 才会提前停止，随后可用**同一参数加 `--resume --stop-after 60000`**续训。不要改变 `--run-id`、初始化 checkpoint 或其他训练参数后再续训；新一轮重训换新的 `--run-id`。

训练时可在另一个终端查看当前变体的 loss 和验证指标（替换路径中的变体名、阶段和 loss 模式）：

```bash
tail -f expe/runs/scratch/viv_gf_e1/seed42/train_loss.csv
tail -f expe/runs/scratch/viv_gf_e1/seed42/history.csv
```

训练器会拒绝在已有输出目录重新开始，恢复前先检查 checkpoint 协议，配对实验还记录初始化 checkpoint 的 SHA256。恢复保存模型、optimizer、scheduler、步数和 best 指标，但没有保存 DataLoader worker 的随机状态；因此是可继续训练，不保证与不中断训练的每一步 crop 完全一致。先前宽度 16 或带 4 个 refinement block 的 E1，以及三窗口系数模型的 checkpoint，均与当前结构不兼容，E1 及所有 A/B/C 配对实验都要按本页命令重新训练。Windows 四卡训练尚未实际跑通，当前只有 CPU 测试；实际运行前需确认 `D:\ResearchCode\S2R-HDR\data\20260818` 和共享 cache 文件存在。

独立 test-200 评估入口：

```bash
python expc/src/evaluate_expc.py --checkpoint expe/runs/a1_a2_e1_matched_4gpu/viv_gf_a1_stats/best.pth --data-root /mnt/d/ResearchCode/S2R-HDR/data/20260818 --output-dir expe/evaluation/viv_gf_a1_stats_test --device cuda:0 --precision fp32
```

边缘/平坦误差、原始系数图和可选单卡 4K 完整前向计时：

```bash
python expe/diagnostics.py --checkpoint expe/runs/a1_a2_e1_matched_4gpu/viv_gf_a1_stats/best.pth --e1-checkpoint expe/runs/scratch/viv_gf_e1/seed42_ddpfix/best.pth --data-root /mnt/d/ResearchCode/S2R-HDR/data/20260818 --output-dir expe/evaluation/viv_gf_a1_stats_validation --split validation --oracle-comparison --save-maps 4 --benchmark-4k --device cuda:0 --precision fp32
```

`--split test` 仅在选定方案后运行。4K 计时把首张输入缩放到 2160x3840，记录 3 次 warmup 后 10 次前向；它不计数据读取，和真实 4K 图像的内容分布不是同一件事。`maps_*.npz` 保存未归一化的系数、候选和权重，便于后续画热力图。测试：`python -m pytest -q expe/tests/test_model.py expe/tests/test_diagnostics.py`。实现侧已完成 CPU 模型、loss、指标与新增外接 Refine/Joint 测试；E1 的 WSL 四卡60K完成情况已通过实际checkpoint和history.csv核对。其余变体的正式训练、独立测试与 4K 单卡速度尚未确认。
