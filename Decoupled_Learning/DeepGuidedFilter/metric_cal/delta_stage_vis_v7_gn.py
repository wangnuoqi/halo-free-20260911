from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ORIGIN_INPUT_DIR = Path("/mnt/d/nuoqi/S2R-HDR/data/test_vivo/testinput")
COARSE_INPUT_DIR = Path("/mnt/d/nuoqi/S2R-HDR/data/test_vivo/df_output")
MODEL_PATH = Path(
    "/mnt/d/nuoqi/halo-free/Decoupled_Learning/DeepGuidedFilter/models_retrain/rf_3stage_best_gtmask_signloss_v7_gn.pth"
)
OUTPUT_ROOT = Path(
    "/mnt/d/nuoqi/halo-free/Decoupled_Learning/DeepGuidedFilter/metric_cal/deltavis_v7"
)

VIS_PERCENTILE = 99.0
IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
GN_GROUPS = 8


def make_norm(num_channels, num_groups=GN_GROUPS):
    groups = min(num_groups, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class Stage1DeltaNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            make_norm(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, base_channels, kernel_size=3, padding=1, bias=False),
            make_norm(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            make_norm(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, kernel_size=1, bias=False),
        )

    def forward(self, x_raw, x_coarse):
        g, _, _ = compute_sobel_grad(x_coarse, normalize=True)
        inp = torch.cat([x_raw, x_coarse, g], dim=1)
        return self.net(inp)


class ABAdjustNet(nn.Module):
    def __init__(self, base_channels=32, a_scale=0.2, b_scale=0.2, use_delta_as_input=True):
        super().__init__()
        self.a_scale = a_scale
        self.b_scale = b_scale
        self.use_delta_as_input = use_delta_as_input

        in_ch = 3 + (1 if use_delta_as_input else 0)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1, bias=False),
            make_norm(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, base_channels, 3, padding=1, bias=False),
            make_norm(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
            make_norm(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 2, 1, bias=True),
        )

    def forward(self, x_raw, x_coarse_k, delta_prev=None):
        g, _, _ = compute_sobel_grad(x_coarse_k, normalize=True)
        if self.use_delta_as_input:
            assert delta_prev is not None
            inp = torch.cat([x_raw, x_coarse_k, g, delta_prev], dim=1)
        else:
            inp = torch.cat([x_raw, x_coarse_k, g], dim=1)

        ab = self.net(inp)
        a_raw = ab[:, 0:1]
        b_raw = ab[:, 1:2]
        a = 1.0 + self.a_scale * torch.tanh(a_raw)
        b = self.b_scale * torch.tanh(b_raw)
        return a, b


class CascadeRefine3Stage(nn.Module):
    def __init__(self, base_channels=32, a_scale=0.2, b_scale=0.2, use_delta_as_input=True):
        super().__init__()
        self.stage1 = Stage1DeltaNet(base_channels=base_channels)
        self.stage2 = ABAdjustNet(
            base_channels=base_channels,
            a_scale=a_scale,
            b_scale=b_scale,
            use_delta_as_input=use_delta_as_input,
        )
        self.stage3 = ABAdjustNet(
            base_channels=base_channels,
            a_scale=a_scale,
            b_scale=b_scale,
            use_delta_as_input=use_delta_as_input,
        )

    def forward(self, x_raw, x_coarse0):
        delta1 = self.stage1(x_raw, x_coarse0)
        x_coarse1 = x_coarse0 + delta1

        a2, b2 = self.stage2(x_raw, x_coarse1, delta1)
        delta2 = a2 * delta1 + b2
        x_coarse2 = x_coarse0 + delta2

        a3, b3 = self.stage3(x_raw, x_coarse2, delta2)
        delta3 = a3 * delta2 + b3
        x_final = x_coarse0 + delta3

        return {
            "delta1": delta1,
            "delta2": delta2,
            "delta3": delta3,
            "final": x_final,
            "a2": a2,
            "b2": b2,
            "a3": a3,
            "b3": b3,
        }


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


def normalize_pair_key(stem: str):
    return stem.split("_")[0]


def collect_images(folder: Path):
    files = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            files[normalize_pair_key(path.stem)] = path
    return files


def read_image(path: Path):
    img = imageio.imread(str(path)).astype(np.float32)
    if img.ndim == 3:
        img = img.mean(axis=2)
    return img


def crop_pair_to_common_shape(a: np.ndarray, b: np.ndarray):
    height = min(a.shape[0], b.shape[0])
    width = min(a.shape[1], b.shape[1])
    return a[:height, :width], b[:height, :width]


def crop_to_even(tensor: torch.Tensor):
    _, _, height, width = tensor.shape
    new_height = height if height % 2 == 0 else height - 1
    new_width = width if width % 2 == 0 else width - 1
    return tensor[:, :, :new_height, :new_width]


def tensor_to_numpy_img(x: torch.Tensor):
    arr = x.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    return arr.astype(np.float32)


def save_float32_tiff(path: Path, img: np.ndarray):
    imageio.imwrite(str(path), img.astype(np.float32))


def to_signed_vis(delta: np.ndarray, percentile: float = VIS_PERCENTILE):
    if delta.ndim == 3:
        delta_vis = delta.mean(axis=2)
    else:
        delta_vis = delta

    abs_max = float(np.percentile(np.abs(delta_vis), percentile))
    if abs_max < 1e-12:
        abs_max = 1.0

    normalized = np.clip(delta_vis / abs_max, -1.0, 1.0)
    colored = plt.get_cmap("seismic")((normalized + 1.0) * 0.5)[..., :3]
    return (colored * 255.0).astype(np.uint8), abs_max


def ensure_dirs():
    for stage_name in ("delta1", "delta2", "delta3"):
        (OUTPUT_ROOT / stage_name).mkdir(parents=True, exist_ok=True)


def build_model(device: torch.device):
    model = CascadeRefine3Stage(base_channels=32, a_scale=0.2, b_scale=0.2, use_delta_as_input=True)
    model = nn.DataParallel(model).to(device)
    state_dict = torch.load(str(MODEL_PATH), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    if not ORIGIN_INPUT_DIR.is_dir():
        raise FileNotFoundError(f"ORIGIN_INPUT_DIR not found: {ORIGIN_INPUT_DIR}")
    if not COARSE_INPUT_DIR.is_dir():
        raise FileNotFoundError(f"COARSE_INPUT_DIR not found: {COARSE_INPUT_DIR}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")

    ensure_dirs()
    origin_map = collect_images(ORIGIN_INPUT_DIR)
    coarse_map = collect_images(COARSE_INPUT_DIR)
    common_names = sorted(set(origin_map) & set(coarse_map))
    if not common_names:
        raise RuntimeError("No matched filenames found between origin input and coarse input")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)

    print(f"[Info] Matched {len(common_names)} image pairs.")
    print(f"[Info] Device: {device}")
    print(f"[Info] Output root: {OUTPUT_ROOT}")

    with torch.no_grad():
        for idx, name in enumerate(common_names, start=1):
            origin_img = read_image(origin_map[name])
            coarse_img = read_image(coarse_map[name])
            origin_img, coarse_img = crop_pair_to_common_shape(origin_img, coarse_img)

            origin_t = torch.from_numpy(origin_img).unsqueeze(0).unsqueeze(0)
            coarse_t = torch.from_numpy(coarse_img).unsqueeze(0).unsqueeze(0)
            origin_t = crop_to_even(origin_t).to(device) - 0.5
            coarse_t = crop_to_even(coarse_t).to(device) - 0.5

            outputs = model(origin_t, coarse_t)

            stats = []
            for stage_name in ("delta1", "delta2", "delta3"):
                delta = tensor_to_numpy_img(outputs[stage_name])
                delta_dir = OUTPUT_ROOT / stage_name
                delta_tiff_path = delta_dir / f"{name}_{stage_name}.tiff"
                delta_png_path = delta_dir / f"{name}_{stage_name}_signed_vis.png"

                save_float32_tiff(delta_tiff_path, delta)
                delta_vis, vis_abs_max = to_signed_vis(delta)
                imageio.imwrite(str(delta_png_path), delta_vis)
                stats.append(
                    f"{stage_name}: min={delta.min():.6f}, max={delta.max():.6f}, "
                    f"vis=[{-vis_abs_max:.6f}, {vis_abs_max:.6f}]"
                )

            print(f"[{idx}/{len(common_names)}] {name} | " + " | ".join(stats))

    print("[Done] Exported delta1 / delta2 / delta3 TIFFs and signed visualization PNGs.")


if __name__ == "__main__":
    main()
