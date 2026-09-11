import os

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

from common_experiment import ExperimentConfig, build_arg_parser, run_experiment


CONFIG = ExperimentConfig(
    name="v1",
    description="Baseline paraminput_freezeb reproduction: raw 2-channel parameter maps + freeze policy B.",
    visible_devices="0,1",
    cond_mode="raw",
    cond_channels=2,
    freeze_mode="b",
    use_front_adapter=False,
)


def main():
    parser = build_arg_parser("DataParallel training for try-freezb v1.")
    args = parser.parse_args()
    run_experiment(args, CONFIG)


if __name__ == "__main__":
    main()
