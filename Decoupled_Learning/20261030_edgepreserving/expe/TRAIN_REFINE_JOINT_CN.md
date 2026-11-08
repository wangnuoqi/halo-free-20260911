# Exp E：E0/E1 Base → 原三阶段 Refine → 整体 Joint

完整参数、当前结果和训练协议差异见 [EXPERIMENT_DETAILS_CN.md](EXPERIMENT_DETAILS_CN.md)。

## E0 Base

E0 是旧 Exp7C coarse（24 通道、dilation=2/4/8/16 的 LRNet + 原 3x3 ConvGuided）在当前协议下的重新训练。新的匹配预算使用双卡每卡16、梯度累积2、有效 batch64、60K更新；输入、128/64 crop、BF16、AdamW、验证划分和 Base loss 与 E1 相同。旧 `seed42_rerun` 权重未对齐，仅作历史参考。

```bash
cd /mnt/d/ResearchCode/halo-free/Decoupled_Learning/20261030_edgepreserving
bash expe/scripts/run_e0_new_matched_ddp_2gpu.sh
```

训练日志在 `expe/runs/scratch/exp7c_e0/seed42_exposure3840000/`；最佳权重为其中的 `best.pth`。Windows 双卡命令由 `python expe/matched_2x2_wintrain.py new_e0` 打印。不要给 E0 加 `--init-lr-checkpoint` 或 `--freeze-lr`。

已读取并核对 E1 完整 checkpoint：保存于 `expe/runs/scratch/viv_gf_e1/seed42_ddpfix/`。`best.pth` 为55000步，验证MSE=3.18309010071971e-05，PSNR=44.97151068042029 dB；`latest.pth`与`final.pth`均为60000步。最终全图验证MSE=3.189749554849186e-05、PSNR=44.962434145029626 dB，与用户日志一致。后续初始化使用该目录的 `best.pth`。

此前 Exp E 的 joint 是 coarse 内部 LRNet 与系数估计一起训练，不含外接三阶段 Refine。本次补齐外接训练链。复用 `exp_b/src/train_exp_b.py` 引擎，并由 `expe/refine_joint.py` 适配 Exp E Base；所有 Exp E coarse 变体（含 E0）都可通过 `--expe-variant` 选择。这里的三阶段 Refine 是继承的下游网络，与 coarse 内部 C1/C2/C3 系数迭代是两组不同实验。

## 运行

```bash
cd /mnt/d/ResearchCode/halo-free/Decoupled_Learning/20261030_edgepreserving
export VARIANT=exp7c_e0
export BASE_CHECKPOINT=expe/runs/scratch/exp7c_e0/seed42_exposure3840000/best.pth
export REFINE_DIR=expe/runs/three_stage/exp7c_e0/seed42_exposure3840000/refine
export JOINT_DIR=expe/runs/three_stage/exp7c_e0/seed42_exposure3840000/joint
bash expe/scripts/run_refine_joint_ddp.sh refine
bash expe/scripts/run_refine_joint_ddp.sh joint
```

也可以使用 `all` 自动串联两个阶段：Refine 成功退出后，脚本才会加载
`refine_best.pth` 并启动 Joint；Refine 失败时不会启动 Joint。

```bash
bash expe/scripts/run_refine_joint_ddp.sh all
```

E1 使用同样的命令，只需将 `VARIANT`、`BASE_CHECKPOINT`、`REFINE_DIR` 和 `JOINT_DIR` 换成 `viv_gf_e1` 的对应目录。

如果完成的 Base 在别处，先将 BASE_CHECKPOINT 换成实际路径。脚本不附加 CUDA 自检，也不自动开始运行；由用户启动。

四卡 NCCL，BF16，每卡 batch16（global64），每卡8 workers（合计32个数据 worker），pin_memory=False，HR crop128、配对LR64，AdamW weight_decay1e-4。数据为原 20260818 LR TIFF / HR input / WLS target。seed42，train内留200张全图验证，每epoch验证；核对 Base checkpoint 的训练/验证ID哈希及数据路径，独立test-200不参与选择。

Refine：冻结完整 Base，只训练原 ThreeStageRefine。默认200 epochs、LR3e-4降到1e-6。Joint：从同一个 Base 及该 Refine 的最佳 checkpoint 加载，解冻全网络，默认200 epochs、LR1e-5降到1e-6。这里200 epochs是新阶段的默认预算，不等同于Base的60K iterations；可以在首次启动前设置 `EPOCHS`，两个阶段各自创建新optimizer/scheduler，不接着Base的末端1e-6学习率跑。

原Refine目标：三个阶段预测相对coarse0的累计delta，目标均为 `GT - coarse0`。各阶段使用 `平坦区L1 + 2×边缘区L1 + 0.2×边缘Sobel向量L1`，边缘由目标delta的梯度经Otsu得到，仅用于训练。三个阶段权重0.1/0.3/0.6。Joint另加 `0.5×smooth_l1(coarse0,GT)`，沿用现有引擎协议。该项与E1训练的动态平衡MSE+Sobel目标不同，后续可单独消融。

E1模型原本接收[0,1] HR/LR；适配器把引擎传来的中心化输入还原，再调用E1，避免重复减0.5。Refine仅包装ThreeStageRefine，关闭find_unused_parameters以避免额外计算图遍历；Joint启用DDP unused parameter处理，以兼容VIVNet上游未使用的reduce_chan_level2权重。验证使用解包模块，避免各rank验证样本数不一致时触发DDP同步问题。

## 输出与恢复

默认输出：

- `expe/runs/three_stage/viv_gf_e1/seed42/refine/{refine_best,refine_latest,refine_final}.pth`
- `expe/runs/three_stage/viv_gf_e1/seed42/joint/{joint_best,joint_latest,joint_final}.pth`

checkpoint为包含 `base.network.*` 和 `refine.*` 的整网 `model_state`，与原coarse checkpoint的 `model` 格式不同；不可直接交给仅支持coarse的 evaluate_expc.py。包含optimizer/scheduler和best指标；先验证更新best，再保存latest，避免旧best值恢复后覆盖历史最佳。

每首次/50步记录rank0批次loss，epoch日志为跨rank汇总平均值：

```bash
tail -f expe/runs/three_stage/viv_gf_e1/seed42/refine/console.log
tail -f expe/runs/three_stage/viv_gf_e1/seed42/refine/train_steps.jsonl
tail -f expe/runs/three_stage/viv_gf_e1/seed42/refine/loss_history.jsonl
```

joint路径将refine替换为joint。已有checkpoint目录拒绝直接覆盖。保持相同配置恢复：

```bash
bash expe/scripts/run_refine_joint_ddp.sh refine --resume
bash expe/scripts/run_refine_joint_ddp.sh joint --resume
```

按epoch保存恢复，epoch内中断后会从最近完成的epoch继续；未保存worker随机状态，不保证逐步crop完全复现。更换Base、Refine或训练超参数应使用新输出目录，可设置 REFINE_DIR、JOINT_DIR、BASE_CHECKPOINT、REFINE_CHECKPOINT、VARIANT。

CPU测试覆盖输入约定、冻结/解冻梯度、初始化与错误variant、Gloo单rank连续两步DDP（含unused参数）、小型配对TIFF数据上Refine→Joint训练及best/latest一致性。未启动四卡GPU正式训练。
