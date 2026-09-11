# try-freezb Experiments

这个目录基于 `/mnt/d/nuoqi/halo-free/Decoupled-params/train_convguided_base_paraminput_freezeb.py` 做 5 个互补实验。

共同约束：

- 训练调用风格保持和原脚本一致：
  - `CUDA_VISIBLE_DEVICES=...`
  - 内部默认使用所有可见 GPU
  - `device=torch.device("cuda:0")`
  - `torch.nn.DataParallel(..., device_ids=list(range(torch.cuda.device_count())), output_device=0)`
- 数据、损失、优化器、StepLR、验证流程都与原脚本保持同一逻辑。
- 单参数预训练权重仍从 `PRETRAINED_PATH` 加载，并兼容第一层输入通道数变化。

## 实验列表

### v1

文件：
- `train_convguided_base_paraminput_freezeb_v1.py`
- `run_v1.sh`

逻辑：
- 原始 `paraminput_freezeb` 复现版
- 参数直接作为 `2` 个常数图拼到输入
- 冻结策略 `B`
  - 开放 `lr[0,1,2,15,16,17,18]`
  - 冻结 `gf`

默认 GPU：
- `0,1`

### v2

文件：
- `train_convguided_base_paraminput_freezeb_v2.py`
- `run_v2.sh`

逻辑：
- 保持冻结策略 `B`
- 但把参数输入从 `2` 个常数图增强成 `8` 个条件图
- 这 8 个条件图来自一个可训练参数编码器：
  - `2 -> 16 -> 8`
  - 输出再 broadcast 到低分辨率空间后与 `x_lr` 拼接

默认 GPU：
- `0,1`

### v3

文件：
- `train_convguided_base_paraminput_freezeb_v3.py`
- `run_v3.sh`

逻辑：
- 继承 `v2` 的 8 通道参数编码
- 冻结策略从 `B` 扩展到 `B+front1`
  - 额外开放 `lr[3,4,5]`
- 目标是测试“再多开一个早期 block”是否能提升 early feature 对参数的适配

默认 GPU：
- `2,3`

### v4

文件：
- `train_convguided_base_paraminput_freezeb_v4.py`
- `run_v4.sh`

逻辑：
- 继承 `v3` 的 8 通道参数编码和 `B+front1` 冻结策略
- 在 `lr[2]` 后插一个轻量 `FrontFiLMAdapter`
- adapter 只做前段条件调制，不放在 tail
- 目标是测试“参数早注入 + early front adapter”是否优于只靠前层解冻

默认 GPU：
- `2,3`

### v5

文件：
- `train_convguided_base_paraminput_freezeb_v5.py`
- `run_v5.sh`

逻辑：
- 继承 `v4` 的 8 通道参数编码和 front FiLM adapter
- 训练策略改成 3 阶段渐进解冻
  - `epoch < 20`：训练参数分支、front adapter 和前段/尾段少量层
  - `20 <= epoch < 50`：解冻整个 `lr` 主干，`gf` 仍冻结
  - `epoch >= 50`：全模型联合微调
- 目标是测试“先稳住条件分支，再逐步放开主干”是否比严格冻结更适合同时学参数泛化和内容泛化

新增模块和插入位置：

- `ParamMapEncoder`
  - 2 个新增全连接层：`Linear(2 -> 16)` 和 `Linear(16 -> 8)`
  - 作用：把 `(lambda, alpha)` 编码成 `8` 维条件向量，再 broadcast 成 `8` 张参数图
  - 插入位置：不直接插在 `lr` 主干内部，而是在输入侧先生成参数图，再与 `x_lr` 拼接后送入 `lr`
  - 对应输入变化：`lr` 分支输入从 `1` 通道灰度图，变成 `1 + 8 = 9` 通道

- `FrontFiLMAdapter`
  - 3 个新增可学习层：`1x1 Conv(24 -> 24)`、`Linear(2 -> 32)`、`Linear(32 -> 48)`
  - 其中 `48 = 24 gamma + 24 beta`，分别对 `24` 个特征通道做缩放和平移
  - 插入位置：`lr` 主干前段，在 `front_adapter_after=2` 处执行，也就是 `self.lr` 顺序模块跑到 `idx == 2` 后，对当前特征做一次 FiLM 调制
  - 计算方式：先由参数生成 `gamma/beta`，再执行 `feat = proj(feat)` 和 `feat * (1 + gamma) + beta`

总新增层数：

- `ParamMapEncoder`：2 层
- `FrontFiLMAdapter`：3 层
- 合计：5 个新增可学习层

数据流可以理解为：

1. `lambda/alpha -> ParamMapEncoder -> 8` 维条件向量
2. `8` 维条件向量扩展成参数图，与 `x_lr` 拼接
3. 拼接后的特征进入 `lr` 主干
4. 在 `lr[2]` 之后插入一次 `FrontFiLMAdapter` 调制前段特征
5. `lr` 主干最后仍输出单通道结果，再送入 `gf` 生成最终图像

和 NAS 版本不同，这里的输出头没有拆成多头控制。`v5` 的主要改进点是“参数怎么进入 backbone”：

- 一次通过 `ParamMapEncoder` 作为条件图输入
- 一次通过 `FrontFiLMAdapter` 作为前层特征调制

三段式训练在 `v5` 里的具体含义：

- stage 0：只先把参数相关新模块和少量 `lr` 层训稳
  - 开放 `param_encoder`
  - 开放 `front_adapter`
  - 开放 `lr[0,1,2,3,4,5,15,16,17,18]`
  - 冻结 `gf`
- stage 1：解冻整个 `lr` 主干，但 `gf` 继续冻结
- stage 2：再解冻 `gf`，全模型联合 finetune

默认 GPU：
- `0,1,2,3`

## 运行方式

### v1

```bash
cd /mnt/d/nuoqi/halo-free/Decoupled-params/try-freezb
./run_v1.sh
```

### v2

```bash
cd /mnt/d/nuoqi/halo-free/Decoupled-params/try-freezb
./run_v2.sh
```

### v3

```bash
cd /mnt/d/nuoqi/halo-free/Decoupled-params/try-freezb
./run_v3.sh
```

### v4

```bash
cd /mnt/d/nuoqi/halo-free/Decoupled-params/try-freezb
./run_v4.sh
```

### v5

```bash
cd /mnt/d/nuoqi/halo-free/Decoupled-params/try-freezb
./run_v5.sh
```

## 常用环境变量

所有 `run_v*.sh` 都支持：

```bash
TRAIN_BATCH_PER_GPU=12
TEST_BATCH=4
TRAIN_WORKERS=12
TEST_WORKERS=2
EPOCHS=80
PIN_MEMORY=1
NO_RESUME=1
```

例如：

```bash
EPOCHS=40 TRAIN_BATCH_PER_GPU=8 TEST_BATCH=2 TRAIN_WORKERS=12 NO_RESUME=1 ./run_v3.sh
```

Windows PowerShell 直跑示例：

```bash
/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command "cd D:\nuoqi\halo-free\Decoupled-params\try-freezb; \$env:CUDA_VISIBLE_DEVICES='0,1,2,3'; conda run --no-capture-output -n yolo_clean python -u .\train_convguided_base_paraminput_freezeb_v5.py --train-batch-per-gpu 12 --test-batch 4 --train-workers 12 --test-workers 2 --epochs 80"
```

## 输出目录

每个实验各自保存：

- checkpoint：
  - `try-freezb/v*/model/`
- tensorboard log：
  - `try-freezb/v*/log_dir/`

checkpoint 命名格式：

- `convguided_filter_full_v*.pth`
- `convguided_filter_last_v*.pth`
- `convguided_filter_best_v*.pth`

## 统一评测

新增文件：

- `eval_try_freezb.py`
- `run_eval_v12.sh`
- `run_eval_v34.sh`

用途：

- `run_eval_v12.sh` 批量测试 `v1` 和 `v2`
- `run_eval_v34.sh` 批量测试 `v3` 和 `v4`

默认 GPU：

- `run_eval_v12.sh` 默认 `CUDA_VISIBLE_DEVICES=0`
- `run_eval_v34.sh` 默认 `CUDA_VISIBLE_DEVICES=2`

直接运行：

```bash
cd /mnt/d/nuoqi/halo-free/Decoupled-params/try-freezb
./run_eval_v12.sh
./run_eval_v34.sh
```

可选环境变量：

```bash
MODEL_KIND=best
BATCH_SIZE=1
NUM_WORKERS=0
SAVE_PREDS=1
```

例如：

```bash
MODEL_KIND=best BATCH_SIZE=1 ./run_eval_v12.sh
MODEL_KIND=best BATCH_SIZE=1 ./run_eval_v34.sh
```

输出：

- 汇总 CSV:
  - `/mnt/d/nuoqi/halo-free/Decoupled-params/try-freezb/eval_outputs/try_freezb_eval_summary.csv`
- 若开启 `SAVE_PREDS=1`，预测图会保存在：
  - `try-freezb/eval_outputs/v1_best/`
  - `try-freezb/eval_outputs/v2_best/`
  - `try-freezb/eval_outputs/v3_best/`
  - `try-freezb/eval_outputs/v4_best/`
Windows launch with stable WSL log viewing:

```bash
bash /mnt/d/nuoqi/halo-free/Decoupled-params/try-freezb/run_v5_win_logged.sh
```

This starts the Windows-side training process, writes logs to `logs/train_v5_*.log`, and tails that log from WSL.
Press `Ctrl-C` to stop tailing only; training continues in Windows unless you kill the PID stored in `logs/train_v5_*.pid`.
