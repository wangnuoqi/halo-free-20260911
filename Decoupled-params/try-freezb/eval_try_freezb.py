import argparse
import csv
import os
import time

import imageio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common_experiment import (
    ExperimentConfig,
    GuidedFilterDataset,
    DeepGuidedFilterVariant,
    calc_psnr,
    crop_to_even,
    seed_everything,
    var_custom_collate,
)


DEFAULT_X_LR_DIR = "/mnt/d/nuoqi/S2R-HDR/data/mixed-lavel-singlehdr-param/testinput_downsampled"
DEFAULT_X_HR_DIR = "/mnt/d/nuoqi/S2R-HDR/data/mixed-lavel-singlehdr-param/testinput"
DEFAULT_LABEL_DIR = "/mnt/d/nuoqi/S2R-HDR/data/mixed-lavel-singlehdr-param/testlabel"
DEFAULT_ROOT = "/mnt/d/nuoqi/halo-free/Decoupled-params/try-freezb"
DEFAULT_OUTPUT_ROOT = os.path.join(DEFAULT_ROOT, "eval_outputs")
DEFAULT_SUMMARY = os.path.join(DEFAULT_OUTPUT_ROOT, "try_freezb_eval_summary.csv")


EXPERIMENT_CONFIGS = {
    "v1": ExperimentConfig(
        name="v1",
        description="Baseline paraminput_freezeb reproduction: raw 2-channel parameter maps + freeze policy B.",
        visible_devices="0,1",
        cond_mode="raw",
        cond_channels=2,
        freeze_mode="b",
        use_front_adapter=False,
    ),
    "v2": ExperimentConfig(
        name="v2",
        description="MLP-like richer parameter encoding via 8 broadcast channels, same freeze policy B.",
        visible_devices="0,1",
        cond_mode="mlp8",
        cond_channels=8,
        freeze_mode="b",
        use_front_adapter=False,
    ),
    "v3": ExperimentConfig(
        name="v3",
        description="Richer 8-channel parameter encoding plus one extra early block unfrozen.",
        visible_devices="2,3",
        cond_mode="mlp8",
        cond_channels=8,
        freeze_mode="b_plus_front1",
        use_front_adapter=False,
    ),
    "v4": ExperimentConfig(
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
    ),
    "v5": ExperimentConfig(
        name="v5",
        description="Progressive unfreezing with richer 8-channel parameter encoding and a front FiLM adapter.",
        visible_devices="0,1,2,3",
        cond_mode="mlp8",
        cond_channels=8,
        freeze_mode="progressive_v5",
        use_front_adapter=True,
        front_adapter_after=2,
        front_adapter_hidden=32,
        front_adapter_scale=0.3,
        progressive_unfreeze_epochs=(20, 50),
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Single-GPU evaluation for try-freezb v1-v5 experiments.")
    parser.add_argument("--variants", nargs="+", default=list(EXPERIMENT_CONFIGS.keys()), choices=list(EXPERIMENT_CONFIGS.keys()))
    parser.add_argument("--model-kind", default="best", choices=["best", "last", "full"])
    parser.add_argument("--x-lr-dir", default=DEFAULT_X_LR_DIR)
    parser.add_argument("--x-hr-dir", default=DEFAULT_X_HR_DIR)
    parser.add_argument("--label-dir", default=DEFAULT_LABEL_DIR)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save-preds", action="store_true")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY)
    return parser.parse_args()


def save_float32_tiff(img, save_path):
    img = img.detach().cpu().numpy()
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    imageio.imwrite(save_path, img.astype("float32"))


def get_device(args):
    if args.cpu or not torch.cuda.is_available():
        return torch.device("cpu"), False
    visible_gpus = torch.cuda.device_count()
    if args.gpu < 0 or args.gpu >= visible_gpus:
        raise ValueError(f"--gpu={args.gpu} is invalid; visible CUDA devices: {visible_gpus}")
    return torch.device(f"cuda:{args.gpu}"), True


def clean_state_dict(checkpoint):
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if any(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def model_filename(variant, model_kind):
    return f"convguided_filter_{model_kind}_{variant}.pth"


def model_dir_for_variant(variant):
    return os.path.join(DEFAULT_ROOT, variant, "model")


def load_model(variant, model_kind, device):
    config = EXPERIMENT_CONFIGS[variant]
    model_path = os.path.join(model_dir_for_variant(variant), model_filename(variant, model_kind))
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    model = DeepGuidedFilterVariant(config)
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = clean_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, model_path


def evaluate_variant(args, variant, device, use_cuda, loader):
    model, model_path = load_model(variant, args.model_kind, device)
    pred_dir = os.path.join(args.output_root, f"{variant}_{args.model_kind}")
    if args.save_preds:
        os.makedirs(pred_dir, exist_ok=True)

    total_psnr = 0.0
    total_mse = 0.0
    total_time = 0.0
    total_images = 0

    with torch.no_grad():
        for batch_x_lr, batch_x_hr, batch_label, batch_params in tqdm(loader, desc=f"Eval {variant}", leave=False):
            batch_x_hr = crop_to_even(batch_x_hr)
            batch_label = crop_to_even(batch_label)
            batch_x_lr = batch_x_lr.to(device)
            batch_x_hr = batch_x_hr.to(device)
            batch_label = batch_label.to(device)
            batch_params = batch_params.to(device)
            batch_x_lr = batch_x_lr - 0.5
            batch_x_hr = batch_x_hr - 0.5

            start_time = time.time()
            output = model(batch_x_lr, batch_x_hr, batch_params)
            if use_cuda:
                torch.cuda.synchronize(device)
            total_time += time.time() - start_time
            total_images += output.size(0)

            psnr, mse = calc_psnr(output, batch_label)
            total_psnr += psnr
            total_mse += mse

            if args.save_preds:
                for i in range(output.size(0)):
                    save_float32_tiff(output[i], os.path.join(pred_dir, f"sample_{total_images - output.size(0) + i:04d}_pred.tiff"))

    return {
        "variant": variant,
        "model_kind": args.model_kind,
        "model_path": model_path,
        "avg_mse": total_mse / len(loader),
        "avg_psnr": total_psnr / len(loader),
        "avg_ms_per_image": (total_time / max(total_images, 1)) * 1000.0,
        "num_samples": len(loader.dataset),
    }


def write_summary(summary_path, rows):
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    fieldnames = ["variant", "model_kind", "avg_mse", "avg_psnr", "avg_ms_per_image", "num_samples", "model_path"]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    seed_everything(42)
    device, use_cuda = get_device(args)
    dataset = GuidedFilterDataset(args.x_lr_dir, args.x_hr_dir, args.label_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=var_custom_collate,
        pin_memory=use_cuda,
        drop_last=False,
    )

    results = []
    for variant in args.variants:
        result = evaluate_variant(args, variant, device, use_cuda, loader)
        results.append(result)
        print(
            f"[RESULT] variant={result['variant']} kind={result['model_kind']} "
            f"mse={result['avg_mse']:.6f} psnr={result['avg_psnr']:.4f} "
            f"ms_per_image={result['avg_ms_per_image']:.2f} samples={result['num_samples']}"
        )

    write_summary(args.summary_path, results)
    print(f"Summary saved to: {args.summary_path}")


if __name__ == "__main__":
    main()
