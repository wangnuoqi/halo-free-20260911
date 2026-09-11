# try-paramadapter

这个目录现在包含 3 条多参数 adapter 路线：

1. `train_convguided_base_paramadapter_film.py`
   - 旧版 24-channel backbone + 单点 FiLM adapter

2. `train_convguided_base_paramadapter_deep_full_v1.py`
   - 参考 `train_convguided_base.py` 的深层特征调制结构
   - 直接全量训练

3. `train_convguided_base_paramadapter_deep_freeze_v5.py`
   - 同样的深层特征调制结构
   - 按 `try-freezb/v5` 风格做 3 阶段渐进解冻

## 数据

训练/测试数据都固定到多参数数据集：

- `mixed-lavel-singlehdr-param/traininput_downsampled`
- `mixed-lavel-singlehdr-param/traininput`
- `mixed-lavel-singlehdr-param/trainlabel`
- `mixed-lavel-singlehdr-param/testinput_downsampled`
- `mixed-lavel-singlehdr-param/testinput`
- `mixed-lavel-singlehdr-param/testlabel`

路径通过脚本内部自动解析成：

- WSL: `/mnt/d/nuoqi/...`
- Windows: `D:\nuoqi\...`

## 深层调制结构

深层版本直接采用 `train_convguided_base.py` 的思路：

- `head`
- `body`（6 个 residual dilated block）
- `pre_inject`
- `inject_conv`
- 参数生成的 `instance_norm weight/bias`
- `out_conv`
- `gf`

区别只在训练策略：

- `deep_full_v1`: 全量训练
- `deep_freeze_v5`: 渐进解冻
  - `epoch < 20`: 只训练调制头和 late inject path
  - `20 <= epoch < 50`: 解冻整个 `lr`
  - `epoch >= 50`: 全模型联合微调

## 默认资源配置

- GPU: `0,1,2,3`
- `train_workers=8`
- `test_workers=8`

内部 `DataParallel` 自动使用所有可见 GPU。

## 运行

WSL:

```bash
cd /mnt/d/nuoqi/halo-free/Decoupled-params/try-paramadapter
./run_train_deep_full_v1.sh
./run_train_deep_freeze_v5.sh
```

Windows PowerShell from WSL:

```bash
wintrain "cd D:\nuoqi\halo-free\Decoupled-params\try-paramadapter; \$env:CUDA_VISIBLE_DEVICES='0,1,2,3'; conda run --no-capture-output -n yolo_clean python -u .\train_convguided_base_paramadapter_deep_full_v1.py --train-batch-per-gpu 12 --test-batch 4 --train-workers 8 --test-workers 8 --epochs 80"
```

```bash
wintrain "cd D:\nuoqi\halo-free\Decoupled-params\try-paramadapter; \$env:CUDA_VISIBLE_DEVICES='0,1,2,3'; conda run --no-capture-output -n yolo_clean python -u .\train_convguided_base_paramadapter_deep_freeze_v5.py --train-batch-per-gpu 12 --test-batch 4 --train-workers 8 --test-workers 8 --epochs 80"
```
