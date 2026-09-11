import argparse
import os
import time
from pathlib import Path


DEFAULT_LR_DIR = Path("/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/down_sample_2_testinput")
DEFAULT_HR_DIR = Path("/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/testinput")
DEFAULT_MODEL_PATH = Path("/mnt/d/nuoqi/halo-free/Decoupled_Learning/total-pipeline-v1/checkpoints/convguided_filter_best_mix.pth")
DEFAULT_OUTPUT_DIR = Path("/mnt/d/nuoqi/halo-free/Decoupled_Learning/20260705_vis/model_outputs")


def build_file_pairs(lr_dir: Path, hr_dir: Path) -> list[tuple[Path, Path]]:
    lr_map = {p.name: p for p in sorted(lr_dir.iterdir()) if p.is_file()}
    hr_map = {p.name: p for p in sorted(hr_dir.iterdir()) if p.is_file()}
    common = sorted(set(lr_map) & set(hr_map))
    if not common:
        raise ValueError(f"No matching files found between {lr_dir} and {hr_dir}")
    return [(lr_map[name], hr_map[name]) for name in common]


def output_path_for_input(output_dir: Path, input_path: Path) -> Path:
    return output_dir / f"{input_path.stem}_pred.tiff"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepGuidedFilter inference and save float32 TIFF outputs.")
    parser.add_argument("--lr-dir", type=Path, default=DEFAULT_LR_DIR, help="Downsampled single-channel input directory.")
    parser.add_argument("--hr-dir", type=Path, default=DEFAULT_HR_DIR, help="Full-resolution single-channel input directory.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Checkpoint path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save model outputs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import cv2
    import imageio.v3 as iio
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm

    class AdaptiveNorm(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.w_0 = nn.Parameter(torch.Tensor([1.0]))
            self.w_1 = nn.Parameter(torch.Tensor([0.0]))
            self.bn = nn.BatchNorm2d(n, momentum=0.999, eps=0.001)

        def forward(self, x):
            return self.w_0 * x + self.w_1 * self.bn(x)

    def weights_init_identity(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            n_out, n_in, h, w = m.weight.data.size()
            if n_out < n_in:
                nn.init.xavier_uniform_(m.weight.data)
                return
            m.weight.data.zero_()
            ch, cw = h // 2, w // 2
            for i in range(n_in):
                m.weight.data[i, i, ch, cw] = 1.0
        elif classname.find("BatchNorm2d") != -1:
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)

    class ConvGuidedFilter(nn.Module):
        def __init__(self, radius=1, norm=nn.BatchNorm2d):
            super().__init__()
            self.box_filter = nn.Conv2d(1, 1, kernel_size=3, padding=radius, dilation=radius, bias=False, groups=1)
            self.conv_a = nn.Sequential(
                nn.Conv2d(2, 32, kernel_size=1, bias=False),
                norm(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=1, bias=False),
                norm(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, kernel_size=1, bias=False),
            )
            self.box_filter.weight.data[...] = 1.0

        def forward(self, x_lr, y_lr, x_hr):
            _, _, h_lrx, w_lrx = x_lr.size()
            _, _, h_hrx, w_hrx = x_hr.size()

            n = self.box_filter(x_lr.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0))
            mean_x = self.box_filter(x_lr) / n
            mean_y = self.box_filter(y_lr) / n
            cov_xy = self.box_filter(x_lr * y_lr) / n - mean_x * mean_y
            var_x = self.box_filter(x_lr * x_lr) / n - mean_x * mean_x
            a = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
            b = mean_y - a * mean_x
            mean_a = F.interpolate(a, (h_hrx, w_hrx), mode="bilinear", align_corners=True)
            mean_b = F.interpolate(b, (h_hrx, w_hrx), mode="bilinear", align_corners=True)
            return mean_a * x_hr + mean_b

    def build_lr_net(norm=AdaptiveNorm, layer=5):
        layers = [
            nn.Conv2d(1, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            norm(24),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for l in range(1, layer):
            layers += [
                nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=2**l, dilation=2**l, bias=False),
                norm(24),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        layers += [
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            norm(24),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(24, 1, kernel_size=1, stride=1, padding=0, dilation=1),
        ]
        net = nn.Sequential(*layers)
        net.apply(weights_init_identity)
        return net

    class DeepGuidedFilter(nn.Module):
        def __init__(self, radius=1, eps=1e-8):
            super().__init__()
            self.lr = build_lr_net()
            self.gf = ConvGuidedFilter(radius, norm=AdaptiveNorm)

        def forward(self, x_lr, x_hr):
            return self.gf(x_lr, self.lr(x_lr), x_hr).clamp(0, 1)

    class GuidedFilterInferenceDataset(Dataset):
        def __init__(self, lr_dir: Path, hr_dir: Path):
            self.pairs = build_file_pairs(lr_dir, hr_dir)

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            lr_path, hr_path = self.pairs[idx]
            x_lr = cv2.imread(str(lr_path), cv2.IMREAD_UNCHANGED).astype("float32")
            x_hr = cv2.imread(str(hr_path), cv2.IMREAD_UNCHANGED).astype("float32")
            if x_lr.ndim != 2 or x_hr.ndim != 2:
                raise ValueError(f"Expected single-channel images, got {lr_path.name}: {x_lr.shape}, {hr_path.name}: {x_hr.shape}")
            return {
                "input_lr": torch.from_numpy(x_lr).unsqueeze(0),
                "input_hr": torch.from_numpy(x_hr).unsqueeze(0),
                "filename": hr_path.name,
                "input_height_hr": x_hr.shape[0],
                "input_width_hr": x_hr.shape[1],
                "input_height_lr": x_lr.shape[0],
                "input_width_lr": x_lr.shape[1],
            }

    def var_custom_collate(batch):
        min_h_hr = min(item["input_height_hr"] for item in batch)
        min_w_hr = min(item["input_width_hr"] for item in batch)
        min_h_lr = min(item["input_height_lr"] for item in batch)
        min_w_lr = min(item["input_width_lr"] for item in batch)

        batch_x_hr = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
        batch_x_lr = torch.zeros(len(batch), 1, min_h_lr, min_w_lr)
        filenames = []

        for idx, item in enumerate(batch):
            batch_x_lr[idx] = item["input_lr"][:, :min_h_lr, :min_w_lr]
            batch_x_hr[idx] = item["input_hr"][:, :min_h_hr, :min_w_hr]
            filenames.append(item["filename"])

        return batch_x_lr, batch_x_hr, filenames

    def crop_to_even(tensor):
        _, _, h, w = tensor.shape
        new_h = h if h % 2 == 0 else h - 1
        new_w = w if w % 2 == 0 else w - 1
        return tensor[:, :, :new_h, :new_w]

    def save_float32_tiff(img, save_path: Path):
        arr = img.detach().cpu().numpy()
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        arr = arr.astype(np.float32)
        iio.imwrite(save_path, arr)

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = DeepGuidedFilter(radius=1, eps=1e-8)
    if device.type == "cuda":
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    dataset = GuidedFilterInferenceDataset(args.lr_dir, args.hr_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=var_custom_collate,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    total_time = 0.0
    total_images = 0

    with torch.no_grad():
        for batch_x_lr, batch_x_hr, filenames in tqdm(loader, desc="Inference", dynamic_ncols=True):
            batch_x_hr = crop_to_even(batch_x_hr)
            batch_x_lr = crop_to_even(batch_x_lr)
            batch_x_lr = batch_x_lr.to(device) - 0.5
            batch_x_hr = batch_x_hr.to(device) - 0.5

            start = time.time()
            outputs = model(batch_x_lr, batch_x_hr)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.time() - start
            total_images += outputs.shape[0]

            for idx, filename in enumerate(filenames):
                save_path = output_path_for_input(args.output_dir, Path(filename))
                save_float32_tiff(outputs[idx], save_path)

    avg_ms = (total_time / max(total_images, 1)) * 1000.0
    print(f"Saved {total_images} model outputs to {args.output_dir}")
    print(f"Average inference time per image: {avg_ms:.2f} ms")


if __name__ == "__main__":
    main()
