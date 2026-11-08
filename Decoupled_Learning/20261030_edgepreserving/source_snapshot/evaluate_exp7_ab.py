import argparse
import importlib.util
import math
import os
import re
from pathlib import Path
import numpy as np


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def files_for(directory):
    return sorted(Path(directory).iterdir())


def numeric_id(path):
    match = re.match(r"(\d+)", path.name)
    return match.group(1) if match else None


def paired_files(lr_dir, origin_dir, label_dir):
    lr = files_for(lr_dir)
    origin = files_for(origin_dir)
    label = files_for(label_dir)
    if len(lr) == len(origin) == len(label):
        return list(zip(lr, origin, label))
    def index(paths):
        return {numeric_id(p): p for p in paths if numeric_id(p) is not None}
    lr_i, origin_i, label_i = index(lr), index(origin), index(label)
    # The test set is defined by origin input + label. If one downsampled
    # input is missing, keep the sample and generate that LR image online.
    ids = sorted(set(origin_i) & set(label_i))
    return [(lr_i.get(i), origin_i[i], label_i[i]) for i in ids]


def downsample_half_long_edge(img):
    from PIL import Image
    h, w = img.shape[:2]
    h -= h % 2
    w -= w % 2
    img = img[:h, :w]
    if h >= w:
        new_h = max(1, h // 2)
        new_w = max(1, int(w * (new_h / h)))
    else:
        new_w = max(1, w // 2)
        new_h = max(1, int(h * (new_w / w)))
    return np.asarray(Image.fromarray(img.astype("float32"), mode="F").resize(
        (new_w, new_h), resample=Image.Resampling.LANCZOS), dtype="float32")


def load_state(model, path, torch):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


def crop_even(x):
    return x[:, :, :x.shape[-2] - x.shape[-2] % 2,
             :x.shape[-1] - x.shape[-1] % 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--lr-dir", required=True)
    parser.add_argument("--origin-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--conv-checkpoint", required=True)
    parser.add_argument("--rf-checkpoint", required=True)
    parser.add_argument("--rf-source", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    import cv2
    import numpy as np
    import torch
    from tqdm import tqdm

    root = Path(__file__).parent
    conv_source = load_module(root / "train_convguided_20260818_l0p500_a1p400.py", "conv_source_exp7")
    rf_source = load_module(args.rf_source, "rf_source_exp7")
    pairs = paired_files(args.lr_dir, args.origin_dir, args.label_dir)
    if not pairs:
        raise RuntimeError("No matched test triples found")
    print(f"Matched test triples: {len(pairs)}")

    device = torch.device("cuda:0")
    conv = conv_source.DeepGuidedFilter(radius=1, eps=1e-8).to(device)
    rf = rf_source.CascadeRefine3Stage(base_channels=32, a_scale=0.2,
                                       b_scale=0.2, use_delta_as_input=True).to(device)
    load_state(conv, args.conv_checkpoint, torch)
    load_state(rf, args.rf_checkpoint, torch)
    conv.eval(); rf.eval()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conv_metrics, rf_metrics = [], []
    with torch.no_grad():
        for index, (lr_path, origin_path, label_path) in enumerate(
                tqdm(pairs, total=len(pairs), desc="Evaluating")):
            origin = cv2.imread(str(origin_path), cv2.IMREAD_UNCHANGED).astype("float32")
            label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED).astype("float32")
            if lr_path is None:
                lr = downsample_half_long_edge(origin)
            else:
                lr = cv2.imread(str(lr_path), cv2.IMREAD_UNCHANGED).astype("float32")
            lr = torch.from_numpy(lr)[None, None].to(device)
            origin = crop_even(torch.from_numpy(origin)[None, None]).to(device)
            label = crop_even(torch.from_numpy(label)[None, None]).to(device)
            coarse = conv(lr - 0.5, origin - 0.5)
            coarse_vis = torch.clamp(coarse + 0.5, 0, 1)
            final = torch.clamp(rf(origin - 0.5, coarse - 0.5)["final"] + 0.5, 0, 1)
            cmse = torch.mean((coarse_vis - label) ** 2).item()
            rmse = torch.mean((final - label) ** 2).item()
            conv_metrics.append(cmse)
            rf_metrics.append(rmse)
            preview = (final[0, 0].cpu().numpy() * 255).clip(0, 255).astype("uint8")
            cv2.imwrite(str(output_dir / f"{index:04d}_{origin_path.stem}_rf.png"), preview)

    cmse = sum(conv_metrics) / len(conv_metrics)
    rmse = sum(rf_metrics) / len(rf_metrics)
    cpsnr = 100 if cmse == 0 else 10 * math.log10(1 / cmse)
    rpsnr = 100 if rmse == 0 else 10 * math.log10(1 / rmse)
    report = (f"Matched test triples: {len(pairs)}\n\n"
              f"ConvGuided\nMSE: {cmse:.8f}\nPSNR: {cpsnr:.4f} dB\n\n"
              f"RF final\nMSE: {rmse:.8f}\nPSNR: {rpsnr:.4f} dB\n")
    (output_dir / "metrics.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
