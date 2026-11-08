# NAFNet WLS baseline

This folder is the grouped entry point for the single-channel NAFNet WLS run.
The original root-level script paths remain available for compatibility with
the active three-GPU run. Links here expose those same scripts and the vendored
NAFNet source without moving files while that job is running.

- `train_nafnet_wls.py`: NCCL/DDP trainer
- `train_nafnet_wls_wintrain.py`: Windows DataParallel trainer
- `wls_data.py`: shared WLS pair reader and crop logic
- `NAFNet/`: vendored NAFNet source and license
- `../runs/nafnet_wls_patch256_nccl_3gpu_bs32/`: current run output

Launch the current three-GPU protocol from the project root with:

```bash
bash baseline/run_nafnet_wls_nccl_3gpu_bs32.sh --resume
```

The wrapper uses the original script path and the existing output directory.
Do not start a second process against that same checkpoint directory.
