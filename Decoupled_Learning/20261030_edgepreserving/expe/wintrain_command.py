"""Print a pasteable WinTrain command for one reproducible Exp E run."""

import argparse
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from expe.model import EXPE_VARIANTS


ROOT = r"D:\ResearchCode\halo-free\Decoupled_Learning\20261030_edgepreserving"
RUNS = ROOT + r"\expe\runs"
DATA = r"D:\ResearchCode\S2R-HDR\data\20260818"
CACHE = ROOT + r"\expc\runs\0818_memmap_4gpu\train_0818_float32.npy"
TRAIN = ROOT + r"\expc\src\train_expc.py"


def command(variant, phase, run_id, seed, stop_after, loss_mode, e1_checkpoint,
            source_checkpoint, resume=False):
    if phase in {"paired", "joint"} and variant == "viv_gf_e1":
        raise ValueError("E1 is trained with --phase scratch")
    if phase == "joint" and loss_mode != "base":
        raise ValueError("Joint initialization needs an explicit same-variant --source-checkpoint for non-base loss")
    if phase != "paired" and loss_mode != "base":
        raise ValueError("Loss ablations use --phase paired")
    if phase == "scratch":
        output = rf"{RUNS}\scratch\{variant}\{run_id}"
        init = ""
    elif phase == "paired":
        output = rf"{RUNS}\paired\{variant}\{loss_mode}\{run_id}"
        source = e1_checkpoint or rf"{RUNS}\scratch\viv_gf_e1\seed42\best.pth"
        init = f" --init-lr-checkpoint {source} --freeze-lr"
    else:
        output = rf"{RUNS}\joint\{variant}\{run_id}"
        source = source_checkpoint or rf"{RUNS}\paired\{variant}\base\seed42\best.pth"
        init = f" --init-model-checkpoint {source}"
    payload = (
        '$env:CUDA_VISIBLE_DEVICES="0,1,2,3"; '
        '$env:PYTHONUNBUFFERED="1"; $env:OMP_NUM_THREADS="1"; '
        f"conda run --no-capture-output -n yolo_clean python -u {TRAIN} "
        f"--parallelism dp --variant {variant} --data-root {DATA} "
        f"--output-dir {output} --cache-path {CACHE}{init} "
        f"--crop-size 128 --batch-size-per-gpu 16 --total-iter 60000 "
        f"--stop-after {stop_after} --val-count 200 --val-every 5000 "
        f"--save-every 100 --num-workers 4 --precision bf16 --lr 3e-4 "
        f"--min-lr 1e-6 --seed {seed} --loss-mode {loss_mode}"
        + (" --resume" if resume else "")
    )
    return "wintrain " + shlex.quote(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(EXPE_VARIANTS), required=True)
    parser.add_argument("--phase", choices=("scratch", "paired", "joint"), required=True)
    parser.add_argument("--run-id", default="seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop-after", type=int, default=60000)
    parser.add_argument("--loss-mode", choices=("base", "edge", "staged", "staged_regularized"), default="base")
    parser.add_argument("--e1-checkpoint", default=None)
    parser.add_argument("--source-checkpoint", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.run_id.isascii() or not args.run_id.replace("_", "").replace("-", "").isalnum():
        parser.error("run-id must contain only ASCII letters, digits, _ or -")
    if args.stop_after < 1 or args.stop_after > 60000:
        parser.error("stop-after must be between 1 and 60000")
    try:
        print(command(args.variant, args.phase, args.run_id, args.seed,
                      args.stop_after, args.loss_mode, args.e1_checkpoint,
                      args.source_checkpoint, args.resume))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
