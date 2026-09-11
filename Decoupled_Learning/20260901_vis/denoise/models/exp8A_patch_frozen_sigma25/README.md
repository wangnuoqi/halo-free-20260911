# Exp8A Patch Frozen Model

这是 `20260621_denoiseretrain/exp8/formal_v2` 中 Exp8A 的 patch 版本冻结模型，
仅使用 Exp8A，不使用 Exp1 或 Exp7 的模型。

```text
source image -> online sigma25 AWGN (no clamp) -> noisy HR
            -> Lanczos-4 half-resolution LR -> Exp8A ConvGuided -> coarse
            -> frozen Exp6D RF -> final
```

模型文件：

- `convguided_patch_sigma25_best.pth`
- `refine_patch_exp6d_sigma25_best.pth`

来源：

- ConvGuided base：`exp8/formal_v2/models/convguided_base_patch128/convguided_best.pth`
- RF bundle：`exp8/formal_v2/runs/exp8A_convguided_frozen_exp6d_patch/convguided_convguided_patch_frozen_exp6d_best.pth`
- patch base：128x128，32 patches/image，80 epochs
- RF：冻结 ConvGuided，Exp6D，batch 32，160 epochs
- BSD68 测试：coarse 27.113882 dB，final 28.818789 dB

BSD68 推理协议：直接枚举 `data_bsds/BSD68` 下的 68 张 clean 图像；每张图同时作为
clean input 的来源和 GT。随后按 Exp8 测试逻辑生成未 clamp 的 sigma25 AWGN，使用 noisy
HR 的 Lanczos-4 半尺寸输入 ConvGuided，再送入冻结的 Exp6D RF。

推理入口：

```bash
bash /mnt/d/nuoqi/halo-free/Decoupled_Learning/20260901_vis/denoise/run_exp8a_inference.sh
```

输出默认保存到 `denoise/outputs/exp8a_patch_frozen/`，分为 `source/`、
`noisy/`、`coarse/`、`final/` 和 `gt/` 五个目录，均为 float32 TIFF。
其中 `source/`、`coarse/`、`final/` 和 `gt/` 在 `[0, 1]`；`noisy/` 保留
协议要求的未 clamp 在线 AWGN 输入。默认使用每张图独立初始化的
`RandomState(0)`。输出指标记录在 `metrics.txt`。
