import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from net.fcnBn import FcnBn
from util.dataset import PairedImageDataset


DEFAULT_DATA_ROOT = Path(r"D:\nuoqi\S2R-HDR\data\mixed_lavel_singlehdr")
DEFAULT_CKPT = Path(__file__).resolve().parent / "checkpoints" / "deepwls_best.pth"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate deepwls and optionally export predictions.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def save_prediction(tensor, output_path):
    array = tensor.squeeze(0).mean(dim=0).clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
    Image.fromarray(array, mode="F").save(output_path)


def main():
    args = parse_args()
    device = torch.device(args.device)
    dataset = PairedImageDataset(
        args.data_root / "testinput",
        args.data_root / "testlabel",
        split="test",
        patch_size=None,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = FcnBn().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    cleaned_state_dict = {
        key.replace("module.", "", 1): value for key, value in state_dict.items()
    }
    model.load_state_dict(cleaned_state_dict, strict=True)
    model.eval()

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    total_mse = 0.0
    with torch.no_grad():
        for inputs, targets, sample_id in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            total_mse += torch.mean((outputs - targets) ** 2).item()

            if args.output_dir is not None:
                save_prediction(outputs[0], args.output_dir / f"{sample_id[0]}-pred.tiff")

    avg_mse = total_mse / max(len(loader), 1)
    print(f"avg_test_mse={avg_mse:.6f}")


if __name__ == "__main__":
    main()
