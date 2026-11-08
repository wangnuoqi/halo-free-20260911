"""Print a Windows WinTrain command for a matched-exposure Exp E cell."""

import argparse
import shlex


ROOT = r"D:\ResearchCode\halo-free\Decoupled_Learning\20261030_edgepreserving"
DATA = r"D:\ResearchCode\S2R-HDR\data\20260818"
RUNS = ROOT + r"\expe\runs"
OLD_RUNS = RUNS + r"\legacy_matched_exposure3840000_v5"


def command(cell: str, phase: str = "base", resume: bool = False) -> str:
    if cell == "new_e0":
        if phase != "base":
            raise ValueError("new_e0 command covers the Base cell only")
        train = ROOT + r"\expc\src\train_expc.py"
        output = RUNS + r"\scratch\exp7c_e0\seed42_exposure3840000"
        cache = ROOT + r"\expc\runs\0818_memmap_4gpu\train_0818_float32.npy"
        arguments = (
            f"{train} --parallelism ddp --variant exp7c_e0 --data-root {DATA} "
            f"--output-dir {output} --cache-path {cache} --crop-size 128 "
            "--batch-size-per-gpu 16 --grad-accumulation 2 --total-iter 60000 "
            "--val-count 200 --val-every 5000 --save-every 100 "
            "--num-workers 4 --precision bf16 --lr 3e-4 --min-lr 1e-6 "
            "--seed 42 --loss-mode base"
        )
    else:
        if cell not in {"old_e0", "old_e1"}:
            raise ValueError(f"unknown cell: {cell}")
        variant = "legacy_matched_e0" if cell == "old_e0" else "legacy_matched_e1"
        root = OLD_RUNS + (r"\e0" if cell == "old_e0" else r"\e1")
        output = root + "\\" + phase
        train = ROOT + r"\expe\train_legacy_matched.py"
        init = ""
        if phase in {"refine", "joint"}:
            init = f" --base-checkpoint {root}\\base\\best.pth"
        if phase == "joint":
            init += f" --refine-checkpoint {root}\\refine\\best.pth"
        arguments = (
            f"{train} --parallelism ddp --variant {variant} --phase {phase} "
            f"--data-root {DATA} --output-dir {output}{init} --crop-size 128 "
            "--batch-size-per-rank 16 --grad-accumulation 2 --total-iter 60000 "
            "--val-count 200 --val-every 5000 --save-every 1000 "
            "--num-workers 4 --seed 42 --lr-base 1e-4 --lr-refine 1e-2 "
            "--lr-joint-coarse 3e-4 --lr-joint-refine 3e-3 --device cuda"
        )
    payload = (
        '$env:CUDA_VISIBLE_DEVICES="0,1"; $env:TORCH_DDP_BACKEND="gloo"; '
        '$env:PYTHONUNBUFFERED="1"; $env:OMP_NUM_THREADS="1"; '
        f"conda run --no-capture-output -n yolo_clean python -u -m torch.distributed.run "
        f"--standalone --nproc_per_node=2 {arguments}"
        + (" --resume" if resume else "")
    )
    return "wintrain " + shlex.quote(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cell", choices=("new_e0", "old_e0", "old_e1"))
    parser.add_argument("--phase", choices=("base", "refine", "joint"), default="base")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        print(command(args.cell, args.phase, args.resume))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
