import argparse
import os
import random
import re
import time

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate parameter-conditioned ConvGuidedFilter.")
    parser.add_argument(
        "--x-lr-dir",
        default="/mnt/d/nuoqi/S2R-HDR/data/mixed-lavel-singlehdr-param/testinput_downsampled",
        help="Low-resolution guidance input directory.",
    )
    parser.add_argument(
        "--x-hr-dir",
        default="/mnt/d/nuoqi/S2R-HDR/data/mixed-lavel-singlehdr-param/testinput",
        help="High-resolution input directory.",
    )
    parser.add_argument(
        "--label-dir",
        default="/mnt/d/nuoqi/S2R-HDR/data/mixed-lavel-singlehdr-param/testlabel",
        help="Optional label directory. Set empty string to disable metrics and label-based param parsing.",
    )
    parser.add_argument(
        "--model-path",
        default="/mnt/d/nuoqi/halo-free/Decoupled-params/model/convguided_filter_best_mix_param_deeplr_injectlate.pth",
        help="Model checkpoint or state_dict path.",
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/d/nuoqi/S2R-HDR/data/mixed-lavel-singlehdr-param/test_refine_input",
        help="Directory to save predicted TIFF files.",
    )
    parser.add_argument(
        "--param-source",
        choices=["label", "hr", "lr", "cli"],
        default="label",
        help="Where to read lambda/alpha from.",
    )
    parser.add_argument("--lambda-val", type=float, default=None, help="Used when --param-source cli.")
    parser.add_argument("--alpha-val", type=float, default=None, help="Used when --param-source cli.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-suffix", default="_pred.tiff")
    parser.add_argument(
        "--device-ids",
        type=int,
        nargs="+",
        default=[0, 1],
        help="CUDA device ids used for DataParallel, matching the training script.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference.",
    )
    return parser.parse_args()


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
        self.box_filter = nn.Conv2d(
            1, 1, kernel_size=3, padding=radius, dilation=radius, bias=False, groups=1
        )
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


class ResidualDilatedBlock(nn.Module):
    def __init__(self, channels, dilation, norm=AdaptiveNorm):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.norm1 = norm(channels)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.norm2 = norm(channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class LRNet(nn.Module):
    def __init__(self, norm=AdaptiveNorm, layer=5):
        super().__init__()
        del layer
        channels = 64
        inject_channels = 64

        self.head = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            norm(channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        dilations = [1, 2, 4, 8, 4, 2]
        self.body = nn.Sequential(
            *[ResidualDilatedBlock(channels, dilation=d, norm=norm) for d in dilations]
        )
        self.pre_inject = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            norm(channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc1_weight = nn.Linear(2, 16)
        self.fc1_bias = nn.Linear(2, 16)
        self.fc2_weight = nn.Linear(16, inject_channels)
        self.fc2_bias = nn.Linear(16, inject_channels)
        self.param_relu = nn.ReLU(inplace=True)
        self.inject_conv = nn.Conv2d(
            channels, inject_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False
        )
        self.out_conv = nn.Conv2d(inject_channels, 1, kernel_size=1, stride=1, padding=0, dilation=1)

        self.apply(weights_init_identity)

    def forward(self, x, params):
        feat = self.head(x)
        feat = self.body(feat)
        feat = self.pre_inject(feat)

        norm_weight = self.fc2_weight(self.param_relu(self.fc1_weight(params)))
        norm_bias = self.fc2_bias(self.param_relu(self.fc1_bias(params)))

        batch_outputs = []
        for n in range(feat.size(0)):
            feat_n = self.inject_conv(feat.narrow(0, n, 1))
            feat_n = F.instance_norm(feat_n, weight=norm_weight[n], bias=norm_bias[n])
            batch_outputs.append(F.leaky_relu(feat_n, negative_slope=0.2, inplace=False))

        feat = torch.cat(batch_outputs, dim=0)
        return self.out_conv(feat)


def build_lr_net(norm=AdaptiveNorm, layer=5):
    return LRNet(norm=norm, layer=layer)


class DeepGuidedFilter(nn.Module):
    def __init__(self, radius=1, eps=1e-8):
        super().__init__()
        del eps
        self.lr = build_lr_net()
        self.gf = ConvGuidedFilter(radius, norm=AdaptiveNorm)

    def forward(self, x_lr, x_hr, params):
        return self.gf(x_lr, self.lr(x_lr, params), x_hr).clamp(0, 1)


def parse_wls_params(path):
    name = os.path.basename(path)
    match = re.search(r"-l([0-9eE+\-.]+)-a([0-9eE+\-.]+)-h", name)
    if match is None:
        raise ValueError(f"Cannot parse lambda/alpha from filename: {name}")
    return float(match.group(1)), float(match.group(2))


class GuidedFilterDataset(Dataset):
    def __init__(self, x_lr_dir, x_hr_dir, label_dir=None, param_source="label", lambda_val=None, alpha_val=None):
        self.x_lr_files = sorted(
            [os.path.join(x_lr_dir, f) for f in os.listdir(x_lr_dir) if os.path.isfile(os.path.join(x_lr_dir, f))]
        )
        self.x_hr_files = sorted(
            [os.path.join(x_hr_dir, f) for f in os.listdir(x_hr_dir) if os.path.isfile(os.path.join(x_hr_dir, f))]
        )
        self.label_files = []
        if label_dir:
            self.label_files = sorted(
                [os.path.join(label_dir, f) for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))]
            )

        self.label_dir = label_dir
        self.param_source = param_source
        self.lambda_val = lambda_val
        self.alpha_val = alpha_val

        if len(self.x_lr_files) != len(self.x_hr_files):
            raise ValueError(f"LR/HR count mismatch: {len(self.x_lr_files)} vs {len(self.x_hr_files)}")
        if self.label_files and len(self.label_files) != len(self.x_hr_files):
            raise ValueError(f"HR/Label count mismatch: {len(self.x_hr_files)} vs {len(self.label_files)}")
        if self.param_source == "label" and not self.label_files:
            raise ValueError("param_source=label requires a valid --label-dir.")
        if self.param_source == "cli" and (self.lambda_val is None or self.alpha_val is None):
            raise ValueError("param_source=cli requires both --lambda-val and --alpha-val.")

        print(
            f"LR: {len(self.x_lr_files)}, HR: {len(self.x_hr_files)}, "
            f"Label: {len(self.label_files)}, Param source: {self.param_source}"
        )

    def __len__(self):
        return len(self.x_hr_files)

    def _get_param_path(self, idx):
        if self.param_source == "label":
            return self.label_files[idx]
        if self.param_source == "hr":
            return self.x_hr_files[idx]
        if self.param_source == "lr":
            return self.x_lr_files[idx]
        return None

    def __getitem__(self, idx):
        x_lr_path = self.x_lr_files[idx]
        x_hr_path = self.x_hr_files[idx]
        label_path = self.label_files[idx] if self.label_files else None

        x_lr = cv2.imread(x_lr_path, cv2.IMREAD_UNCHANGED).astype("float32")
        x_hr = cv2.imread(x_hr_path, cv2.IMREAD_UNCHANGED).astype("float32")
        label = None
        if label_path is not None:
            label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED).astype("float32")

        if self.param_source == "cli":
            lambda_val, alpha_val = self.lambda_val, self.alpha_val
        else:
            lambda_val, alpha_val = parse_wls_params(self._get_param_path(idx))

        sample = {
            "input_lr": torch.from_numpy(x_lr).unsqueeze(0).unsqueeze(0),
            "input_hr": torch.from_numpy(x_hr).unsqueeze(0).unsqueeze(0),
            "input_params": torch.tensor([lambda_val, alpha_val], dtype=torch.float32),
            "filename": os.path.splitext(os.path.basename(x_hr_path))[0],
            "input_height_hr": x_hr.shape[0],
            "input_width_hr": x_hr.shape[1],
            "input_height_lr": x_lr.shape[0],
            "input_width_lr": x_lr.shape[1],
        }
        if label is not None:
            sample["target_img"] = torch.from_numpy(label).unsqueeze(0).unsqueeze(0)
        return sample


def var_custom_collate(batch):
    min_h_hr = min(item["input_height_hr"] for item in batch)
    min_w_hr = min(item["input_width_hr"] for item in batch)
    min_h_lr = min(item["input_height_lr"] for item in batch)
    min_w_lr = min(item["input_width_lr"] for item in batch)

    has_label = "target_img" in batch[0]
    batch_x_hr = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
    batch_x_lr = torch.zeros(len(batch), 1, min_h_lr, min_w_lr)
    batch_params = torch.zeros(len(batch), 2)
    batch_target_images = torch.zeros(len(batch), 1, min_h_hr, min_w_hr) if has_label else None
    filenames = []

    for idx, item in enumerate(batch):
        batch_x_lr[idx] = item["input_lr"][:, :, :min_h_lr, :min_w_lr]
        batch_x_hr[idx] = item["input_hr"][:, :, :min_h_hr, :min_w_hr]
        batch_params[idx] = item["input_params"]
        filenames.append(item["filename"])
        if has_label:
            batch_target_images[idx] = item["target_img"][:, :, :min_h_hr, :min_w_hr]

    return batch_x_lr, batch_x_hr, batch_target_images, batch_params, filenames


def crop_to_even(tensor):
    _, _, h, w = tensor.shape
    new_h = h if h % 2 == 0 else h - 1
    new_w = w if w % 2 == 0 else w - 1
    return tensor[:, :, :new_h, :new_w]


def calc_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    psnr = 100 if mse == 0 else 10 * np.log10(1.0 / mse)
    return psnr, mse


def save_float32_tiff(img, save_path):
    img = img.detach().cpu().numpy()
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    imageio.imwrite(save_path, img.astype(np.float32))


def build_runtime(args):
    if args.cpu:
        return torch.device("cpu"), False, []

    visible_gpus = torch.cuda.device_count()
    if visible_gpus <= 0:
        return torch.device("cpu"), False, []

    if visible_gpus <= max(args.device_ids):
        raise RuntimeError(
            f"Requested GPUs {args.device_ids}, but only {visible_gpus} CUDA device(s) are visible. "
            "Please set CUDA_VISIBLE_DEVICES correctly or pass --cpu."
        )

    return torch.device(f"cuda:{args.device_ids[0]}"), True, args.device_ids


def load_model(model_path, device, use_cuda=False, device_ids=None):
    if device_ids is None:
        device_ids = []
    model = DeepGuidedFilter(radius=1, eps=1e-8)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    is_dp_state_dict = any(key.startswith("module.") for key in state_dict.keys())
    if is_dp_state_dict and use_cuda and len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
    elif is_dp_state_dict:
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}

    if use_cuda:
        model = model.to(device)

    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    args = parse_args()
    seed_everything(42)

    label_dir = args.label_dir if args.label_dir else None
    os.makedirs(args.output_dir, exist_ok=True)

    device, use_cuda, device_ids = build_runtime(args)
    model = load_model(args.model_path, device, use_cuda=use_cuda, device_ids=device_ids)

    dataset = GuidedFilterDataset(
        x_lr_dir=args.x_lr_dir,
        x_hr_dir=args.x_hr_dir,
        label_dir=label_dir,
        param_source=args.param_source,
        lambda_val=args.lambda_val,
        alpha_val=args.alpha_val,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=var_custom_collate,
        pin_memory=use_cuda,
        drop_last=False,
    )

    criterion = torch.nn.MSELoss(reduction="mean")
    test_psnr = 0.0
    test_mse = 0.0
    total_time = 0.0
    total_images = 0
    has_label = label_dir is not None

    with torch.no_grad():
        for test_x_lr, test_x_hr, test_label, test_params, filenames in tqdm(loader, desc="Testing"):
            test_x_hr = crop_to_even(test_x_hr)
            if test_label is not None:
                test_label = crop_to_even(test_label)

            test_x_lr = test_x_lr.to(device)
            test_x_hr = test_x_hr.to(device)
            test_params = test_params.to(device)
            if test_label is not None:
                test_label = test_label.to(device)

            test_x_lr = test_x_lr - 0.5
            test_x_hr = test_x_hr - 0.5

            start_time = time.time()
            test_output = model(test_x_lr, test_x_hr, test_params)
            if use_cuda:
                torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            total_images += test_output.size(0)

            if test_label is not None:
                _ = criterion(test_output, test_label)
                psnr, mse = calc_psnr(test_output, test_label)
                test_mse += mse
                test_psnr += psnr

            for i, fname in enumerate(filenames):
                save_path = os.path.join(args.output_dir, f"{fname}{args.save_suffix}")
                save_float32_tiff(test_output[i], save_path)

    avg_time_per_image = total_time / max(total_images, 1)
    print(f"Saved predictions to: {args.output_dir}")
    print(f"Average Inference Time per Image: {avg_time_per_image * 1000:.2f} ms")
    if has_label:
        avg_test_mse = test_mse / len(loader)
        avg_test_psnr = test_psnr / len(loader)
        print(f"Test Finished - Average PSNR: {avg_test_psnr:.2f} dB, Average MSE: {avg_test_mse:.6f}")


if __name__ == "__main__":
    main()
