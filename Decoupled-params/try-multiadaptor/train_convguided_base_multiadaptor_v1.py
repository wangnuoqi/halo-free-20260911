import os

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from common_experiment import ExperimentConfig, build_arg_parser, run_experiment


CONFIG = ExperimentConfig(
    name="v1",
    description="Multi-adapter residual conditioning across the whole lr trunk on 4 GPUs.",
    visible_devices="0,1,2,3",
    cond_channels=8,
    adapter_hidden=12,
    adapter_scale=0.2,
    adapter_bottleneck=8,
    freeze_mode="multi_adapter_b",
    adapter_positions=(2, 5, 8, 11, 14, 17),
)


def main():
    parser = build_arg_parser("DataParallel training for try-multiadaptor v1.")
    args = parser.parse_args()
    run_experiment(args, CONFIG)


if __name__ == "__main__":
    main()
