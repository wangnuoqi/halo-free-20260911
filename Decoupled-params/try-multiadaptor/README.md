# try-multiadaptor Experiments

这个目录用于测试“广而轻”的多层 adapter 条件调制方案。

当前实现：

- `train_convguided_base_multiadaptor_v1.py`
- `run_v1.sh`
- `common_experiment.py`

## v1 逻辑

- 输入条件使用 `mlp8` 参数编码
- 在 `lr` 主干的多个位置插入条件化残差 adapter
  - 默认位置：`lr[2], lr[5], lr[8], lr[11], lr[14], lr[17]`
- 每个 adapter 形式为：
  - `1x1 down -> ReLU -> 1x1 up`
  - 再叠加由参数生成的 `scale/shift`
- 主干大部分卷积保持冻结
- 默认仍只开放少量前后层，但所有 adapter 和参数编码器都参与训练
- `gf` 默认冻结

## 默认 GPU

- `0,1,2,3`

内部 `DataParallel` 会自动使用所有可见 GPU。

## 输出目录

- checkpoint：
  - `try-multiadaptor/v1/model/`
- tensorboard log：
  - `try-multiadaptor/v1/log_dir/`

## 运行方式

WSL:

```bash
cd /mnt/d/nuoqi/halo-free/Decoupled-params/try-multiadaptor
./run_v1.sh
```

Windows PowerShell from WSL:

```bash
/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command "cd D:\nuoqi\halo-free\Decoupled-params\try-multiadaptor; \$env:CUDA_VISIBLE_DEVICES='0,1,2,3'; conda run --no-capture-output -n yolo_clean python -u .\train_convguided_base_multiadaptor_v1.py --train-batch-per-gpu 12 --test-batch 4 --train-workers 12 --test-workers 2 --epochs 80"
```
