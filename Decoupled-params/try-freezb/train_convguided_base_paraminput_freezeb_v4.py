import os

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2,3")

from common_experiment import ExperimentConfig, build_arg_parser, run_experiment


CONFIG = ExperimentConfig(
    name="v4",
    description="Richer 8-channel parameter encoding, one extra early block unfrozen, plus a front FiLM adapter.",
    visible_devices="2,3",
    cond_mode="mlp8",
    cond_channels=8,
    freeze_mode="b_plus_front1",
    use_front_adapter=True,
    front_adapter_after=2,
    front_adapter_hidden=32,
    front_adapter_scale=0.3,
)


def main():
    parser = build_arg_parser("DataParallel training for try-freezb v4.")
    args = parser.parse_args()
    run_experiment(args, CONFIG)


if __name__ == "__main__":
    main()
