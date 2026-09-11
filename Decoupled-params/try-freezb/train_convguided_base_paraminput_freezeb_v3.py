import os

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2,3")

from common_experiment import ExperimentConfig, build_arg_parser, run_experiment


CONFIG = ExperimentConfig(
    name="v3",
    description="Richer 8-channel parameter encoding plus one extra early block unfrozen.",
    visible_devices="2,3",
    cond_mode="mlp8",
    cond_channels=8,
    freeze_mode="b_plus_front1",
    use_front_adapter=False,
)


def main():
    parser = build_arg_parser("DataParallel training for try-freezb v3.")
    args = parser.parse_args()
    run_experiment(args, CONFIG)


if __name__ == "__main__":
    main()
