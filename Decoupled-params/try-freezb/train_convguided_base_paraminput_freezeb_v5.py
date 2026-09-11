import os

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from common_experiment import ExperimentConfig, build_arg_parser, run_experiment


CONFIG = ExperimentConfig(
    name="v5",
    description="Progressive unfreezing: front adapter warmup, then full lr trunk, then full model finetune on 4 GPUs.",
    visible_devices="0,1,2,3",
    cond_mode="mlp8",
    cond_channels=8,
    freeze_mode="progressive_v5",
    use_front_adapter=True,
    front_adapter_after=2,
    front_adapter_hidden=32,
    front_adapter_scale=0.3,
    progressive_unfreeze_epochs=(20, 50),
)


def main():
    parser = build_arg_parser("DataParallel progressive-unfreeze training for try-freezb v5.")
    args = parser.parse_args()
    run_experiment(args, CONFIG)


if __name__ == "__main__":
    main()
