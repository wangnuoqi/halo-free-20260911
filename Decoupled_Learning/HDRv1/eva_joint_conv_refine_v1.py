import os
import time

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def save_float32_tiff(img, save_path):
    """
    Save float32 tiff without normalization.
    """
    img = img.detach().cpu().numpy()
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.float32)
    imageio.imwrite(save_path, img)


def crop_to_even(tensor):
    _, _, h, w = tensor.shape
    new_h = h if h % 2 == 0 else h - 1
    new_w = w if w % 2 == 0 else w - 1
    return tensor[:, :, :new_h, :new_w]


def compute_sobel_grad(x, eps=1e-6, normalize=False):
    sobel_x = torch.tensor(
        [[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]],
        device=x.device,
        dtype=x.dtype,
    )
    sobel_y = torch.tensor(
        [[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]],
        device=x.device,
        dtype=x.dtype,
    )
    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)
    g = torch.sqrt(gx * gx + gy * gy + eps)
    if normalize:
        g = g / (g.amax(dim=(2, 3), keepdim=True) + eps)
    return g, gx, gy


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


def build_lr_net(norm=AdaptiveNorm, layer=5):
    layers = [
        nn.Conv2d(1, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(24),
        nn.LeakyReLU(0.2, inplace=True),
    ]

    for l in range(1, layer):
        layers += [
            nn.Conv2d(
                24, 24, kernel_size=3, stride=1, padding=2**l, dilation=2**l, bias=False
            ),
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


class EdgeAwareRefineNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, kernel_size=1, bias=False),
        )

    def forward(self, x_raw, x_coarse):
        g, _, _ = compute_sobel_grad(x_coarse, normalize=True)
        inp = torch.cat([x_raw, x_coarse, g], dim=1)
        delta = self.net(inp)
        out = x_coarse + delta
        return out


class JointHDRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.convguided = DeepGuidedFilter(radius=1, eps=1e-8)
        self.refine = EdgeAwareRefineNet(in_channels=3)

    def forward(self, x_lr, x_hr):
        coarse = self.convguided(x_lr, x_hr)
        refined = self.refine(x_hr, coarse)
        return coarse, refined


class GuidedFilterDataset(Dataset):
    def __init__(self, x_lr_dir, x_hr_dir, label_dir):
        self.x_lr_files = sorted([os.path.join(x_lr_dir, f) for f in os.listdir(x_lr_dir)])
        self.x_hr_files = sorted([os.path.join(x_hr_dir, f) for f in os.listdir(x_hr_dir)])
        self.label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])
        print(
            f"LR: {len(self.x_lr_files)}, HR: {len(self.x_hr_files)}, Label: {len(self.label_files)}"
        )

    def __len__(self):
        return min(len(self.x_lr_files), len(self.x_hr_files), len(self.label_files))

    def __getitem__(self, idx):
        x_lr_path = self.x_lr_files[idx]
        x_lr = cv2.imread(x_lr_path, cv2.IMREAD_UNCHANGED).astype("float32")
        x_hr = cv2.imread(self.x_hr_files[idx], cv2.IMREAD_UNCHANGED).astype("float32")
        label = cv2.imread(self.label_files[idx], cv2.IMREAD_UNCHANGED).astype("float32")

        input_hr_h, input_hr_w = x_hr.shape
        input_lr_h, input_lr_w = x_lr.shape
        filename = os.path.splitext(os.path.basename(x_lr_path))[0]

        return {
            "input_lr": torch.from_numpy(x_lr).unsqueeze(0).unsqueeze(0),
            "input_hr": torch.from_numpy(x_hr).unsqueeze(0).unsqueeze(0),
            "target_img": torch.from_numpy(label).unsqueeze(0).unsqueeze(0),
            "filename": filename,
            "input_height_hr": input_hr_h,
            "input_width_hr": input_hr_w,
            "input_height_lr": input_lr_h,
            "input_width_lr": input_lr_w,
        }


def var_custom_collate(batch):
    min_h_hr = min([item["input_height_hr"] for item in batch])
    min_w_hr = min([item["input_width_hr"] for item in batch])
    min_h_lr = min([item["input_height_lr"] for item in batch])
    min_w_lr = min([item["input_width_lr"] for item in batch])

    batch_x_lr = torch.zeros(len(batch), 1, min_h_lr, min_w_lr)
    batch_x_hr = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
    batch_target_images = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
    filenames = []

    for idx, item in enumerate(batch):
        x_lr_crop = item["input_lr"][:, :, :min_h_lr, :min_w_lr]
        x_hr_crop = item["input_hr"][:, :, :min_h_hr, :min_w_hr]
        label_crop = item["target_img"][:, :, :min_h_hr, :min_w_hr]

        batch_x_lr[idx] = x_lr_crop
        batch_x_hr[idx] = x_hr_crop
        batch_target_images[idx] = label_crop
        filenames.append(item["filename"])

    return batch_x_lr, batch_x_hr, batch_target_images, filenames


def calc_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    psnr = 100 if mse == 0 else 10 * np.log10(1 / mse)
    return psnr, mse


# Paths
# Test data
x_lr_test = "/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/down_sample_2_testinput"
x_hr_test = "/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/testinput"
label_test = "/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/testlabel"

# Model and output
model_path = (
    "/mnt/d/nuoqi/halo-free/Decoupled_Learning/HDRv1/models_retrain/"
    "joint_convguided_refine_best_v1.pth"
)
output_dir = (
    "/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/result/joint_convguided_refine_v1"
)
os.makedirs(output_dir, exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = JointHDRNet()
model = nn.DataParallel(model).to(device)

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()


test_dataset = GuidedFilterDataset(
    x_lr_dir=x_lr_test, x_hr_dir=x_hr_test, label_dir=label_test
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=var_custom_collate,
    pin_memory=True,
    drop_last=False,
)


test_mse = 0
test_psnr = 0
total_time = 0
total_images = 0

with torch.no_grad():
    for test_x_lr, test_x_hr, test_label, filenames in tqdm(
        test_loader, desc="Testing"
    ):
        test_x_lr = crop_to_even(test_x_lr)
        test_x_hr = crop_to_even(test_x_hr)
        test_label = crop_to_even(test_label)

        test_x_lr = test_x_lr.to(device)
        test_x_hr = test_x_hr.to(device)
        test_label = test_label.to(device)

        test_x_lr = test_x_lr - 0.5
        test_x_hr = test_x_hr - 0.5

        start_time = time.time()
        _, refined = model(test_x_lr, test_x_hr)
        end_time = time.time()

        refined_vis = torch.clamp(refined + 0.5, 0, 1)

        batch_time = end_time - start_time
        total_time += batch_time
        total_images += test_x_lr.size(0)

        for i, fname in enumerate(filenames):
            save_float32_tiff(refined_vis[i], f"{output_dir}/{fname}_pred.tiff")

        psnr, mse = calc_psnr(refined_vis.detach(), test_label)
        test_mse += mse
        test_psnr += psnr

avg_test_mse = test_mse / len(test_loader)
avg_test_psnr = test_psnr / len(test_loader)
avg_infer_time = total_time / total_images

print(
    f"Test Finished - Average PSNR: {avg_test_psnr:.2f} dB, "
    f"Average MSE: {avg_test_mse:.6f}"
)
print(f"Average inference time per image: {avg_infer_time * 1000:.2f} ms")
