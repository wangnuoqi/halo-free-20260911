# FFDNet -> Refine bridge

This directory now uses the official-author KAIR PyTorch FFDNet release:
`pytorch_official/network_ffdnet.py`,
`pytorch_official/basicblock.py`, and
`pytorch_official/models/ffdnet_gray.pth`. They are from
https://github.com/cszn/KAIR and its v1.0 release. KAIR states that the FFDNet
weights were converted from MatConvNet. No MATLAB executable is required.

The cache generator follows KAIR `main_test_ffdnet.py`: a grayscale 15-layer
network, sigma map `25/255`, replication padding for odd dimensions, and the
official deterministic NumPy test-noise convention.

```bash
cd /mnt/d/nuoqi/halo-free/Decoupled_Learning/20260621_denoiseretrain
PY=/home/nuoqi/miniconda3/envs/halofree/bin/python
ROOT=/mnt/d/nuoqi/halo-free/data_bsds
OUT=third_party/FFDNet/refine_bridge/runs/sigma25
$PY third_party/FFDNet/refine_bridge/generate_cache.py \
  --train-root "$ROOT/BSDS 400" --test-root "$ROOT/BSD68" \
  --out "$OUT/cache.pt" --device cpu
$PY third_party/FFDNet/refine_bridge/train_refine.py \
  --cache "$OUT/cache.pt" --variant groupnorm --epochs 80 \
  --out "$OUT/groupnorm" --device cpu
$PY third_party/FFDNet/refine_bridge/train_refine.py \
  --cache "$OUT/cache.pt" --variant original --epochs 80 \
  --out "$OUT/original" --device cpu
```

`metrics.json` reports the frozen FFDNet BSD68 baseline and refine PSNR.

To replace the old invalid FFDNet cache and rerun only the two FFDNet refine
experiments through Windows four-GPU `wintrain`:

```bash
wintrain '$env:CUDA_VISIBLE_DEVICES="0,1,2,3"; $env:PYTHONUNBUFFERED="1"; conda run --no-capture-output -n yolo_clean python -u D:\nuoqi\halo-free\Decoupled_Learning\20260621_denoiseretrain\third_party\run_refine_four_gpu_sequential.py --data-root D:\nuoqi\halo-free\data_bsds --run-root D:\nuoqi\halo-free\Decoupled_Learning\20260621_denoiseretrain\third_party\refine_runs\bsds400_bsd68_sigma25 --epochs 80 --batch-size 4 --device cuda:0 --jobs ffdnet'
```
