# Exp B 实验交接

## 目标

验证只替换 LR predictor 内部 dilation block 后的效果。ConvGuided、LR predictor 外部接口与三阶段逐像素仿射 refine 递推都保留。

## 训练次序

1. base：从随机初始化训练 VIV-block LR predictor + ConvGuided。
2. refine：加载并冻结 base_best.pth，训练原始三阶段 Refine。
3. joint：加载 base 最优权重和对应 refine 最优权重，解冻全模型联合训练。

三个阶段都从 Exp B 的新初始化开始；没有加载 Exp A、C0/exp7C、NAFNet 或 VIVNet checkpoint。

## 已完成

- src/model_exp_b.py：VIV-inspired 内部 block、原 ConvGuided 和三阶段 refine。
- src/data_exp_b.py：按实际文件名配对三组单通道 TIFF；训练 crop 为 HR/GT 128、LR 64。
- src/train_exp_b.py：base/refine/joint、NCCL DDP、保存 best/latest/final、断点协议校验和 JSONL loss。
- src/evaluate_exp_b.py：200 张测试图全图推理，保存指标，不保存预测图。
- run_sequence_ddp.sh：依次启动三个阶段；另外三个单阶段启动脚本可单独运行。
- 实际数据目录检查：train 2003 组、test 200 组、HR/LR 尺寸比例 2。
- CPU 单步 smoke test：base、refine、joint 均能完成训练、验证和 checkpoint 保存。
- 四卡 smoke test：4 张 RTX 3090、DDP/NCCL、base 阶段单步训练通过；完成 all-reduce、验证和 rank 0 checkpoint 保存，torchrun 正常退出（exit code 0）。测试使用 batch/rank=1、val_count=2，仅验证分布式链路，不代表正式训练配置或模型效果。
- NCCL 清理异常现在会作为 warning 打印，不会覆盖训练阶段原有的异常或已保存结果。
- base、refine、joint 三阶段各 200 epochs 正式训练已完成，best/latest/final checkpoint 和 loss history 均存在。
- 2026-11-04 已在独立 test-200 上完成 FP32 全图评估；base coarse0 为 38.309 dB，refine final 为 36.030 dB，joint final 为 37.725 dB。结果和 Exp7C/NAFNet/VIVNet 参考值见 `evaluation/test_20260818/summary.json`。

## 运行位置

默认 run root：

    exp_b/runs/exp_b_vivblock_20261102/

每阶段目录分别保存：

    base/base_{latest,best,final}.pth
    refine/refine_{latest,best,final}.pth
    joint/joint_{latest,best,final}.pth

测试命令、workers 调整方法和恢复方法见 README_CN.md。

## 当前结论和下一步

- Exp B 的 joint best 在训练验证中心 crop 上约 41.5 dB，但独立 test-200 final 只有 37.725 dB；refine best final 为 36.030 dB，均低于 base coarse0 的 38.309 dB。
- 当前三阶段 refine 在独立测试集上造成退化，模型泛化表现不理想。先检查 train/validation/test 的数据分布和输出误差，再决定 refine 递推或训练目标是否需要调整。
- 尚无同训练协议的原始 LR block 对照，因此不能把 Exp B 与 Exp7C 的差距直接归因于 VIV-inspired block。

已有比较记录见 ../exp_a/evaluation/comparison/README.md：NAFNet 49.434 dB、VIVNet 48.992 dB、旧 C0/exp7C 47.114 dB。这些是历史参考结果，不是 Exp B 结果。
