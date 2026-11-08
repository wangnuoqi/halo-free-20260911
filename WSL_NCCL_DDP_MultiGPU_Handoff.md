# WSL 四卡 NCCL DDP 可复用经验

本文从第 3 节开始记录在 WSL2 + NVIDIA GPU + PyTorch NCCL DDP 环境中排查多卡训练问题的经验。重点针对一种常见现象：`nvidia-smi` 能看到多张卡且显存空闲，但某个训练脚本无法启动，而另一个脚本可以继续四卡训练。

## 3. 先区分 CUDA 初始化、NCCL 通信和模型显存

遇到 `CUDA error: out of memory` 时，先看 traceback 所在阶段，不要直接把所有错误都叫作显存不够。

### 3.1 CUDA 设备初始化阶段

典型错误：

```text
Unexpected error from cudaGetDeviceCount()
Error 2: out of memory
torch.cuda.is_available() == False
```

此时还没有加载模型、数据或 loss，说明 PyTorch 在创建 CUDA context 或枚举 GPU 时失败。`nvidia-smi` 显示的 FB memory 空闲，并不能证明 WSL 的 GPU virtual-address、BAR1 或 context 资源也正常。

### 3.2 NCCL process group 阶段

典型位置：

```python
dist.init_process_group(backend="nccl")
```

如果错误发生在这里，CUDA context 已经创建成功，问题才进入 NCCL 初始化、rank 间通信或 GPU 拓扑层。

### 3.3 模型 forward/backward 阶段

如果 traceback 出现在：

```text
model(...)
loss.backward()
```

才是通常意义上的模型激活或中间 tensor 显存不足。

### 3.4 DataLoader pin-memory 阶段

典型错误：

```text
Caught RuntimeError in pin memory thread
CUDA error: out of memory
```

这通常发生在 validation 或取 batch 阶段。WSL/PyTorch 的 pin-memory worker 可能尝试建立额外的 device-side mapping。模型训练已经完成并不代表 validation 一定能通过。

## 4. 启动 DDP 前的最小检查

先停止同一批 GPU 上的其它多卡训练和监控程序，再检查进程：

```bash
pgrep -af 'torchrun|torch.distributed.run|train_ddp.py|train_dp.py'
```

检查显存和计算进程：

```bash
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

然后只做 CUDA 自检，不启动训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -c \
'import torch; print("available =", torch.cuda.is_available()); print("count =", torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]'
```

四卡 DDP 的最低通过条件是：

```text
available = True
count = 4
```

如果输出 `False 4`，说明设备数量可以被枚举，但 CUDA context 初始化失败；这时不要重复启动训练。

## 5. WSL CUDA context 异常的处理顺序

当 `nvidia-smi` 正常、显存空闲，但 PyTorch CUDA 自检失败时，按以下顺序处理：

1. 正常停止其它训练和 `nvitop` 等 GPU 监控程序。
2. 在 Windows PowerShell 执行：

   ```powershell
   wsl --shutdown
   ```

3. 重新打开 WSL，重新激活 conda 环境。
4. 再执行第 4 节的 CUDA 自检。
5. 如果仍失败，按 `Win + Ctrl + Shift + B` 重置 Windows 图形驱动。
6. 仍失败时重启 Windows，再重新检查。

WSL 内核日志可用于确认 GPU virtual-address 层是否报错：

```bash
dmesg | rg -i 'dxg|nvidia|cuda|gpu|oom'
```

例如：

```text
dxgkio_reserve_gpu_va: Ioctl failed: -75
```

表示 WSL 的 dxg 虚拟 GPU 层在保留 GPU virtual address 时失败。这个问题不一定与 FB 显存占用相关。

## 6. 为什么一个脚本能四卡，另一个脚本不能

需要完整对照两个脚本的 CUDA 初始化顺序，不能只看它们都使用了 NCCL。

已验证的 VivNet 初始化路径是：

```python
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.device_count()
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")
```

一个容易触发问题的路径是：

```python
if not torch.cuda.is_available():
    raise RuntimeError(...)
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")
```

在异常的 WSL 状态下，可能出现：

```text
torch.cuda.device_count()  -> 4
torch.cuda.is_available()  -> False
```

前一种脚本会继续绑定 rank 的 GPU，后一种脚本会在 NCCL 初始化前主动退出。因此不要把 `torch.cuda.is_available()` 当作唯一的 DDP 前置检查；应检查可见设备数量并让每个 rank 尽早绑定自己的 GPU。

推荐的初始化模板：

```python
local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
visible_devices = torch.cuda.device_count()
if visible_devices < world:
    raise RuntimeError(
        f"requested {world} ranks, but only {visible_devices} CUDA devices are visible"
    )
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")
device = torch.device("cuda", local_rank)
```

每个 rank 使用：

```python
torch.cuda.manual_seed(seed)
```

避免在初始化早期使用会触碰所有可见卡的：

```python
torch.cuda.manual_seed_all(seed)
```

## 7. VivNet 四卡设置中哪些是关键的

VivNet 的四卡脚本使用：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3
OMP_NUM_THREADS=1
torch.distributed.run --standalone --nproc_per_node=4
```

训练设置包括：

```text
每卡 batch: 48
patch: 128x128
precision: BF16
num_workers: 4
pin_memory: False
```

这些设置并不都影响 DDP 初始化：

- `CUDA_VISIBLE_DEVICES` 和 `nproc_per_node` 决定 rank 与可见 GPU 数量；
- 先 `set_device(local_rank)` 再初始化 NCCL 是关键路径；
- BF16、patch 大小和 batch 主要影响模型运行阶段显存；
- `pin_memory=False` 减少 WSL host-to-device pinning 相关资源压力；
- `find_unused_parameters=True` 只适用于存在未参与反向传播参数的模型，不是 CUDA 初始化修复。

## 8. WSL 下 DataLoader 的建议

对于容易出现 pin-memory OOM 的 WSL 多卡任务，优先使用：

```python
DataLoader(
    ..., 
    pin_memory=False,
    persistent_workers=True,
)
```

数据仍然可以按原方式移动到 GPU：

```python
batch = batch.to(device, non_blocking=True)
```

`non_blocking=True` 在 pageable host memory 上不会产生错误，只是通常不能获得 pinned-memory 的异步传输收益。稳定性优先时不要强行打开 `pin_memory=True`。

## 9. 四卡 NCCL 启动模板

在 CUDA 自检通过且其它四卡任务已停止后：

```bash
cd /path/to/experiment

CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_MODULE_LOADING=LAZY \
OMP_NUM_THREADS=1 \
torchrun \
  --standalone \
  --nproc_per_node=4 \
  --master_port=29561 \
  training/train_ddp.py \
  --config configs/train.yaml
```

启动后先观察：

```text
DDP world=4
```

然后确认第一个 epoch 的训练和 validation 都完成，并且生成了 checkpoint。只看到训练 batch 进度不够，因为 validation 仍可能在 pin-memory 或验证 forward 阶段失败。

## 10. 错误分层速查

| 症状 | 位置 | 处理方向 |
|---|---|---|
| `torch.cuda.is_available() == False`，`Error 2` | CUDA context 初始化 | 停止并发任务，`wsl --shutdown`，重置驱动 |
| `device_count()==4` 但 `is_available()==False` | WSL lazy init | 对齐 VivNet 的逐 rank 初始化顺序 |
| `ProcessGroupNCCL` 报错 | NCCL 建组/通信 | 查 rank、端口、拓扑和并发任务 |
| `pin memory thread` OOM | DataLoader 取 batch | 关闭 `pin_memory`，减少 workers |
| `model()` 或 `backward()` OOM | 模型激活显存 | AMP、patch、batch 或模型规模 |
| `IndexError` 空 source index | 数据分支验证 | 空索引必须是 `torch.long`，并在使用前检查 `numel()` |
| `SIGTERM`/`SignalException 15` | 外部终止 | 检查调度器、父进程或手动 Ctrl-C，不等同于 CUDA OOM |

## 11. 可复用的最短操作流程

```text
停止其它多卡任务
    -> nvidia-smi
    -> 四卡 torch.cuda 自检
    -> 失败则 wsl --shutdown
    -> 必要时重置 Windows 驱动/重启 Windows
    -> 确认 available=True, count=4
    -> 使用逐 rank CUDA 初始化的 DDP 脚本
    -> DataLoader 使用 pin_memory=False
    -> 验证第一个 epoch 完整结束并保存 checkpoint
```

最后要区分两个事实：VivNet 过去成功运行四卡 DDP，证明这台机器和 WSL 配置具备四卡能力；某次新任务启动失败，通常只能证明当时的初始化顺序、并发上下文或 WSL 驱动状态不适合该任务，不能据此判断 NCCL 永久不兼容。

