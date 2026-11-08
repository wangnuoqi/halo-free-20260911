# Exp E：多窗口与逐阶段仿射估计实验交接

训练结果、历史 Exp7C 与同协议 E0/E1 对照细节见 [EXPERIMENT_DETAILS_CN.md](EXPERIMENT_DETAILS_CN.md)。

## Scope（范围）

研究基础模型从半分辨率预测 `y_lr` 重建高分辨率结果时，如何估计、传递和反复更新局部仿射参数 `a,b`。Exp E 的 A/B/C 先只比较 coarse 输出；之后已另接原三阶段 Refine 与 Joint 链。`q = a * guide_hr + b` 是保留的重建形式；学习到的 `a,b` 不宣称等于经典 guided filter 的解析系数。

Exp E 现在包含在统一协议下重训的旧 Exp7C coarse 对照 E0、VIVNet U-Net 的 E1 基线，以及逐步叠加的 A1/A2/B/C 系数实验。E0 使用原 24 通道、四个 dilation（2/4/8/16）的 LRNet 和原 3x3 ConvGuided；只替换训练输入、crop、batch、优化器、预算和验证协议。E1 保留项目内 VIVNet 基线宽度 32、三个尺度各 1 个 block 的 U-Net 主干，不使用末端 4 个 refinement block。这里的“多尺度 box”指**同一半分辨率网格上的不同统计窗口**，不是新增四分之一分辨率 LRNet 分支。

E 系列 2×2 以已完成的四卡 E1 新配方 Base 的 global batch 64、60,000 次更新、384 万样本呈现为 coarse 统一预算。待训三格的 WSL 顺序入口是 `expe/scripts/run_e_2x2_ddp_2gpu.sh`：GPU 0、1 上用 NCCL DDP，每 rank 8 workers；E0 新配方跑 Base、Refine、Joint，E0 旧配方也跑三阶段，E1 旧配方只跑 Base。Base 双卡每 rank batch 16、累积 2；E0 新配方 Refine/Joint 双卡每 rank 32、200 epochs，与现有 E1 新配方下游的 global batch 64 和 epoch 数对应；旧配方各阶段每 rank 16、累积 2、60K 更新。所有 Base 使用同一 128 HR/64 LR patch 和 train/validation 划分。完成后脚本验证四格协议，仅归档旧 `legacy_matched_patch128*` 配方目录；旧 E0 新配方但暴露量不匹配的 Base/Refine/Joint、已完成的四卡 E1 三阶段及 A 系列均保留。旧配方结果写入 `legacy_matched_exposure3840000_v5`。

最终独立测试统一使用中心 512×512 HR crop、配对 256×256 LR。E1 Base/Refine/Joint 与历史原 Exp7C 的 test-200 同尺寸结果及训练协议差异见 `expe/evaluation/test_200_512/README_CN.md`；更完整的参数、训练输入和 E0/E1 对照见 `expe/EXPERIMENT_DETAILS_CN.md`。历史 Exp7C 与 E1 的 training protocol 未对齐，不可作为仅改变 LRNet 的因果消融。

## Current Status（当前状态）

- **已完成且作为基准：** E1 新配方 Base checkpoint；四卡每卡 16、60K 更新、384 万样本呈现。
- **待重训：** E0 新配方三阶段、E0 旧配方三阶段、E1 旧配方 Base。原 E0 `seed42_rerun` 及历史下游权重属于未对齐结果，但保留为历史对照，不纳入匹配的 2×2 结果。
- **待运行：** 新 A 系列五格按 E1 的训练样本呈现预算训练；双卡每卡16、累积2，对齐 E1 四卡每卡16的有效 global batch64及60K optimizer steps。旧固定统计 A1/A2 及结果已归档，固定 box3/box9 单窗口仍保留。B/C 保持原有独立方案。E1 U-Net 对 `y_lr` 或最终系统质量的因果贡献仍需同协议原 LRNet 对照确认。
- **待运行：** 旧配方 E0/E1 配对训练；两格使用与 E1 新配方相同的 Base 暴露预算。
- **已验证的背景：** 原 ConvGuided 在 LR 上用 3x3 可学习 box 算均值、协方差和方差，以 1x1 网络根据协方差和方差估计 `a`，令 `b = mean_y - a * mean_x`，再分别双线性上采样 `a,b`。因此现有 1x1 系数头并非完全没有邻域信息；邻域统计已由 box 提供。

## Verified Artifacts（已核实文件）

- `README_CN.md`：实验变体、训练协议和 WinTrain 命令。
- `model.py`：旧 Exp7C E0 LRNet、半分辨率 VIVNet U-Net、原 GF 基线、多 box 仿射估计、Otsu/梯度输入、HR 系数融合和迭代更新；没有外接三阶段 Refine。
- `losses.py`、`checkpoint.py`：loss 模式、从 E1 checkpoint 加载 LRNet、从同变体完整 checkpoint 开始 joint 微调的入口。
- `diagnostics.py`：全图目标边缘/平坦误差、少量系数图数组、预测/GT-LR 对照和可选 4K 单卡计时入口；尚无正式运行产物。
- `tests/test_model.py`：形状、梯度、候选分配、Otsu、阶段更新和 checkpoint 初始化测试；通过测试不等于正式训练完成。
- `matched_2x2_wintrain.py`：生成 E0 新配方及旧配方 E0/E1 的 Windows 双卡匹配预算命令；`wintrain_command.py` 保留其他 E 系数变体的旧入口。
- `../expc/src/models.py`、`../expc/src/train_expc.py`、`../expc/src/evaluate_expc.py`：`viv_gf_*` 的模型注册、训练与独立 test 评估入口。
- `../exp_b/src/model_exp_b.py`：当前 ConvGuided 的统计、系数及上采样公式。
- `../source_snapshot/experiments/exp7C_joint_fidelity_20260818.py`：旧 Refine 的逐阶段图像残差更新及 edge-guided loss；其中 Refine 的 `a,b` 作用于上一轮图像残差，并非本实验 guided filter 的 `a,b`。

仓库布局不是 handoff skill 收集器默认的 `experiments/vNN` 布局。收集器报告这些标准目录缺失；上面的状态来自实际 `expe/` 文件、训练入口和目录检查，不能把收集器的 `missing` 误读成源码不存在。

## Reproduction（复现协议）

1. E1 新配方 Base 已完成，作为 384 万样本呈现基准。旧 E0 `seed42_rerun` 虽有60K checkpoint，但双卡运行的暴露量未对齐；不得作为已完成的对照格，也不能从其 checkpoint 续训到新目录。
2. 在 WSL 执行 `bash expe/scripts/run_e_2x2_ddp_2gpu.sh`，按 E0 新配方三阶段、E0 旧配方三阶段、E1 旧配方 Base 的顺序运行。各 Base 都按有效 batch64、60K 更新从头训练；三格使用同一 20260818 配对 LR TIFF / HR input / WLS target、128/64 crop、seed42 的 train/validation split。脚本仅在全部 checkpoint 通过校验后归档旧 `legacy_matched_patch128*` 目录；旧 E0 暴露量不匹配的训练结果和四卡 E1 三阶段继续保留。

3. 新 A 系列五格按 E1 的有效 batch64和60K步预算从头联合训练，LRNet 不冻结，入口见 `LEARNED_KERNEL_EXPERIMENTS_CN.md`。其双卡梯度累积保证总样本呈现量匹配；sampler 分片、裁剪序列及动态 loss 权重不保证逐步梯度完全相同。固定单窗口和现有 B/C 的 paired 方案仍用 `--init-lr-checkpoint .../viv_gf_e1/best.pth --freeze-lr`；B/C 当前沿用旧固定统计分支，不能直接视作新 A2 的后续阶段。
4. `diagnostics.py --oracle-comparison --e1-checkpoint .../viv_gf_e1/best.pth` 在同一个预测 `y_lr` 上并列测双线性、原 GF 和新模块，并用 `area` 下采样 GT 产生的 GT-LR 重复。GT-LR 只用于诊断，不可用于正式推理。最终方案才测独立 test-200；模型选择只看训练划出的 200 张全图验证集。
5. 训练器在恢复前核对协议，并记录初始化 checkpoint 的 SHA256；新启动遇到已有协议、history、训练 loss 或 checkpoint 会拒绝覆盖。每个变体都在自己的 `train_loss.csv` 记录训练 loss、pixel、gradient、学习率（首次、每 100 step 和结束时），`history.csv` 记录每 5K 的全图验证。`base` 模式的 pixel 是 MSE、gradient 是缩放前 Sobel 向量 L1；edge/staged 模式仍使用各自的 Charbonnier/有限差分定义。只有显式短跑时才需要 `--resume --stop-after 60000` 从同目录 `latest.pth` 继续，保留 optimizer/scheduler 与 best 指标。随机 crop 的 DataLoader worker 状态未保存，续训不保证逐步等同于一次性跑满；精确重新比较时每个变体重新从头训练。先前宽度 16 或带 4 个 refinement block 的 E1，以及三窗口系数模型与当前结构不兼容，不能继续使用旧 checkpoint。

## Design and Decision Context（设计与决策依据）

| 阶段 | 逐步加入的能力 | 必须保留的对照 / 判别问题 |
| --- | --- | --- |
| E0 | 原 LRNet + 原 3x3 ConvGuided。 | 当前协议下重训原结构，和 E1 配对。 |
| E1 | 半分辨率 VIVNet U-Net 估计 `y_lr`，仍接原 GF。`guide_hr` 仍是 HR 原图；`y_lr` 提供目标趋势。 | 单独判断 LRNet 结构是否有效，不能与系数改动混在一起归因。 |
| A1 | 在同一 LR 网格保留 E1 完整可学习 3x3 统计分支，增加独立可学习 9x9 分支并逐像素融合 `a,b`；另设两个窗口直接预测 `a,b` 的对照。 | `viv_gf_a1_stats`、`viv_gf_a1_direct` 与 `viv_gf_a1_fixed_boxes`；固定双窗口旧 global32 结果归档，新协议重训纳入串行链；固定单窗口 `viv_gf_a1_box3/box9` 另作消融。 |
| A2 | 分别给统计版和直接预测版 A1 的门控加入 HR 原图与预测 `y_lr` 的梯度及 Otsu 特征，汇总到 LR 网格。 | `viv_gf_a2_stats_edge`、`viv_gf_a2_direct_edge` 与各自 A1 配对。旧固定统计 A2 已归档。 |

### A1 补充消融：自由预测 b 的结果与后续修正

**已完成，但当前设计不足以单独判断自由 `b` 的效果：**`viv_gf_a1_stats_freeb` 在 `expe/runs/a1_a2_e1_matched_2gpu_accum2/viv_gf_a1_stats_freeb/` 完成 60K 更新，内部 validation 最终/最佳 PSNR 为 **42.9308 dB**，MSE 为 `5.09238e-05`，记录见该目录的 `history.csv`、`protocol.json` 和 `final.pth`。同一 A1 训练组的 `viv_gf_a1_stats` 最佳 validation PSNR 为 **45.3692 dB**（`expe/runs/a1_a2_e1_matched_2gpu_accum2/viv_gf_a1_stats/history.csv`）。该差值是内部 validation 比较，不是独立 test 结论。

检查实现后发现，当前 `freeb` branch 输入仍只有 `covariance`、`variance`，最后一层改为直接输出 `[a,b]`；它没有把 `mean_x`、`mean_y` 输入 `b` 的预测路径。局部均值能表征亮度水平，而协方差/方差主要表征局部共同变化和起伏，因此这版自由 `b` 可能缺少估计亮度偏置所需的信息。此外，新 `b` 输出通道零初始化，初始 `b=0`，并不等于 A1 的 `mean_y-a*mean_x`。所以当前比较同时改变了 `b` 的约束、`b` 可用输入和初始化函数，**不能据此断言自由预测 `b` 本身不如均值关系约束**。

**未来待做：**修正自由 `b` 对照，让 `b` 预测路径可以访问对应窗口的 `mean_x`、`mean_y`，同时保留 3×3/9×9 statistics branch 和原 gate。优先让 `a` 继续走 A1 的原预测路径，只给独立 `b` head 增加均值输入，避免额外改变 `a`；再与 `viv_gf_a1_stats` 在相同训练协议下比较。报告 branch 输出、`b` 的空间/数值分布及最终 validation/test 图像指标。只有修正后的对照仍显著落后，才支持“均值关系提供了有效约束”的解释。
| B | HR 像素根据原图、`y_lr` 的目标趋势和各候选的可信度，选择不同窗口及邻近 LR 位置的系数，再形成 HR `a,b`。 | 与双线性系数插值比较；重点检查目标边界两侧的系数串扰。此前固定规则的边缘插值几乎没有收益，B 的假设是可学习、目标相关的选择。 |
| C1 / C2 | 从 `(a_0,b_0,q_0)` 出发，HR stage 同时重估 `a,b`，每次由新系数计算 `q_k = a_k * guide_hr + b_k`；先一阶段，再两阶段，三阶段仅作深度对照。 | `viv_gf_c1`、`viv_gf_c2`、`viv_gf_c3` 比较深度；`viv_gf_c2_aonly` 检验只更新 `a`。`b` 可以承担修正，但需报告它与 `a` 的实际贡献。 |
| L | 在固定结构上比较 GT 边缘加权、`q_0/q_1/q_2` 的逐阶段监督，以及弱系数更新稳定约束。 | E1-C2 的结构对照先共用原 coarse 的 MSE + 动态平衡 Sobel `base` loss；其他模式当前使用不同像素/梯度定义，需作为整体 loss 替换实验报告，不能将差值单独归因于新增监督。 |

多 box 的物理依据是：小窗口更容易定位窄边界，大窗口可在稳定区域汇集更多统计信息，但也更容易跨边缘。1x1 分配头负责决定各位置相信哪个窗口。B 解决的是这些 LR 系数如何传到具体 HR 像素；C 解决的是看到上一轮结果后是否能重新估计得更好。这三步有不同的判别目标。

各阶段输出以图像重建为主监督，不构造唯一的 `a_gt,b_gt`：在平坦区域，不同 `a,b` 可能给出几乎相同的输出。L 的弱约束只用于避免异常大幅更新或无规律振荡，不应预先禁止 `b` 修正。跨窗口、跨阶段也主要比较输出图像，不强求系数数值相同。训练时的 GT Otsu 边缘可用于 loss；推理时进入模型的边缘只能从 HR 输入和预测 `y_lr` 计算。

## Pitfalls and Constraints（风险与约束）

- E1 已有 Base/Refine/Joint 的 test-200 结果；E0 当前没有完整训练或质量结果，因此不能声称 E1 优于 E0。需要完成 E0 同协议训练后再比较。
- Otsu 使用复制边界的 Sobel 梯度及 256-bin 阈值，训练 crop 与整图推理按当前输入范围各自计算。阈值仍可能随 crop/整图变化；要检查漂移，并保留连续梯度对照。推理不得读取 GT 边缘。
- C 的 HR 多阶段计算可能主导 4K 耗时。参数量小不代表延迟小，须测单卡完整前向并与 E0/E1 比较；多卡 DataParallel 的异常延迟不用于结构优劣判断。
- 单凭最终输出 loss 无法唯一识别 `a,b`，因此不能把 PSNR 增长直接解释为每个系数都更接近某个物理真值。需要同时查看系数图、权重图、边缘误差和 `a` / `b` 消融。
- 旧 GT-LR oracle 记录的协议文字与脚本中心化处理存在待核对之处；重做 oracle 诊断时必须确认 `x_lr`、`y_lr`、`guide_hr` 的数值域和中心化约定一致。
- 恢复训练与选择 best checkpoint 时应以验证完成后保存的 `best_mse` 为准，避免旧版“验证前保存 latest”造成 best 覆盖问题；复用当前训练器前核对 checkpoint 元数据。

## Open Questions（待解问题）

- 在同协议训练下，E1 的 VIVNet LR predictor 是否优于 E0 原 LRNet？E0 完成前仍未回答。
- A1 收益来自多 box、额外统计量，还是参数量增加？需匹配单窗口宽度与计算量对照。
- 自由预测 `b` 是否需要显式的局部均值输入？现有 `viv_gf_a1_stats_freeb` validation 为42.9308 dB，但 `b` head 只看到 covariance/variance 且从零输出开始，不能用于判断自由 `b` 与均值约束的优劣；需要完成上述修正版对照。
- A2 的 Otsu 二值图在训练 crop 与整图推理间是否稳定，是否比连续梯度提供额外信息？
- B 的改进是否主要发生在目标边界，而非放大原图中应被抑制的纹理？
- C 的第二阶段是否仍有收益，其 4K 单卡延迟是否可接受？
- 固定 LRNet 有效的 GF 模块，在 joint 微调后是否仍有效，是否改变 `y_lr` 的分布？

## Next Actions（下一步）

1. 为 E0 使用新的独立 run-id 从头训练60K，期间检查每5K全图验证、checkpoint 和训练 loss CSV。
2. E0 Base 完成后训练它自己的 frozen-Base Refine，再从同一 E0 Base + Refine 初始化 Joint 并解冻全部参数。
3. 对齐后比较 E0/E1 的 Base、Refine final、Joint coarse 和 Joint final；validation 用于选 checkpoint，独立 test-200 只做最终报告。
4. A 系列五个变体按 E1 Base 的有效 batch64和60K更新预算从头训练；双卡通过累积2个每卡16的 micro-batch 实现。固定双窗口旧结果是 global32，仅作历史参考。固定单窗口及现有 B/C 仍可用同一 E1 `best.pth` 冻结 LR predictor 做 paired 对照。结构优选后做 loss 消融与 4K 单卡速度诊断。
5. 修正并重跑 A1 自由 `b` 补充实验：让独立 `b` head 得到局部均值信息，保持 `a` 预测路径、gate 和训练配方不变；现有 `stats_freeb` 结果只作为输入不足版本的 preliminary 记录。
