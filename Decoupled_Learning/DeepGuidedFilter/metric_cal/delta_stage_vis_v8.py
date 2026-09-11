from collections import OrderedDict
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
    "/mnt/d/nuoqi/halo-free/Decoupled_Learning/DeepGuidedFilter/models_retrain/rf_3stage_best_gtmask_signloss_v8.pth"
)
OUTPUT_ROOT = Path(
    "/mnt/d/nuoqi/halo-free/Decoupled_Learning/DeepGuidedFilter/metric_cal/deltavis_v8"
)

VIS_PERCENTILE = 99.0
IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
NORM_TYPE = "group"
GN_GROUPS = 8

MODEL_CFG = {
    "base_channels": 32,
    "a_scale": 0.2,
    "b_scale": 0.2,
    "use_delta_as_input": True,
    "norm_type": NORM_TYPE,
    "gn_groups": GN_GROUPS,
    "sign_prior_scale": 0.0,
}

# 每个中间量均导出:
# 1) float32 tiff (保留数值, 便于后处理)
# 2) 可读 png (便于快速查看)
OUTPUT_SPECS = OrderedDict(
    [
        ("delta1", {"mode": "signed"}),
        ("delta2", {"mode": "signed"}),
        ("delta3", {"mode": "signed"}),
        ("coarse1", {"mode": "image_shifted"}),
        ("coarse2", {"mode": "image_shifted"}),
        ("final", {"mode": "image_shifted"}),
        ("stage1_sign_logits", {"mode": "signed"}),
        ("stage1_sign_map", {"mode": "signed"}),
        ("stage1_mag_map", {"mode": "unit"}),
        ("a2", {"mode": "around_one"}),
        ("b2", {"mode": "signed"}),
        ("a3", {"mode": "around_one"}),
        ("b3", {"mode": "signed"}),
    ]
)


def make_norm(num_channels, norm_type="batch", gn_groups=8):
    norm = norm_type.lower()
    if norm in ("batch", "bn", "batchnorm"):
        return nn.BatchNorm2d(num_channels)
    if norm in ("group", "gn", "groupnorm"):
        groups = min(gn_groups, num_channels)
        while groups > 1 and (num_channels % groups) != 0:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class Stage1DeltaNet(nn.Module):
    def __init__(
        self,
        in_channels=5,
        base_channels=32,
        delta_scale=0.25,
        sign_prior_scale=0.0,
        norm_type="batch",
        gn_groups=8,
    ):
        super().__init__()
        self.delta_scale = float(delta_scale)
        self.sign_prior_scale = float(sign_prior_scale)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            make_norm(16, norm_type=norm_type, gn_groups=gn_groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, base_channels, kernel_size=3, padding=1, bias=False),
            make_norm(base_channels, norm_type=norm_type, gn_groups=gn_groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            make_norm(base_channels, norm_type=norm_type, gn_groups=gn_groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 2, kernel_size=1, bias=True),
        )

    def forward(self, x_raw, x_coarse):
        _, gx, gy = compute_sobel_grad(x_coarse, normalize=False)
        gx = gx / (gx.abs().amax(dim=(2, 3), keepdim=True) + 1e-6)
        gy = gy / (gy.abs().amax(dim=(2, 3), keepdim=True) + 1e-6)
        raw_delta = x_raw - x_coarse
        inp = torch.cat([x_raw, x_coarse, raw_delta, gx, gy], dim=1)

        stage1_raw = self.net(inp)
        mag_raw = stage1_raw[:, 0:1]
        sign_raw = stage1_raw[:, 1:2]

        if self.sign_prior_scale != 0.0:
            sign_logits = sign_raw + self.sign_prior_scale * raw_delta
        else:
            sign_logits = sign_raw
        sign_map = torch.tanh(sign_logits)
        mag_map = self.delta_scale * torch.sigmoid(mag_raw)
        delta = sign_map * mag_map
        return {
            "delta": delta,
            "sign_logits": sign_logits,
            "sign_map": sign_map,
            "mag_map": mag_map,
        }


class ABAdjustNet(nn.Module):
    def __init__(
        self,
        base_channels=32,
        a_scale=0.2,
        b_scale=0.2,
        use_delta_as_input=True,
        norm_type="batch",
        gn_groups=8,
    ):
        super().__init__()
        self.a_scale = a_scale
        self.b_scale = b_scale
        self.use_delta_as_input = use_delta_as_input

        in_ch = 3 + (1 if use_delta_as_input else 0)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1, bias=False),
            make_norm(16, norm_type=norm_type, gn_groups=gn_groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, base_channels, 3, padding=1, bias=False),
            make_norm(base_channels, norm_type=norm_type, gn_groups=gn_groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
            make_norm(base_channels, norm_type=norm_type, gn_groups=gn_groups),
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
    def __init__(
        self,
        base_channels=32,
        a_scale=0.2,
        b_scale=0.2,
        use_delta_as_input=True,
        norm_type="batch",
        gn_groups=8,
        sign_prior_scale=0.0,
    ):
        super().__init__()
        self.stage1 = Stage1DeltaNet(
            base_channels=base_channels,
            sign_prior_scale=sign_prior_scale,
            norm_type=norm_type,
            gn_groups=gn_groups,
        )
        self.stage2 = ABAdjustNet(
            base_channels=base_channels,
            a_scale=a_scale,
            b_scale=b_scale,
            use_delta_as_input=use_delta_as_input,
            norm_type=norm_type,
            gn_groups=gn_groups,
        )
        self.stage3 = ABAdjustNet(
            base_channels=base_channels,
            a_scale=a_scale,
            b_scale=b_scale,
            use_delta_as_input=use_delta_as_input,
            norm_type=norm_type,
            gn_groups=gn_groups,
        )

    def forward(self, x_raw, x_coarse0):
        stage1_out = self.stage1(x_raw, x_coarse0)
        delta1 = stage1_out["delta"]
        x_coarse1 = x_coarse0 + delta1

        a2, b2 = self.stage2(x_raw, x_coarse1, delta1)
        delta2 = a2 * delta1 + b2
        x_coarse2 = x_coarse0 + delta2

        a3, b3 = self.stage3(x_raw, x_coarse2, delta2)
        delta3 = a3 * delta2 + b3
        x_final = x_coarse0 + delta3

        return {
            "delta1": delta1,
            "coarse1": x_coarse1,
            "delta2": delta2,
            "coarse2": x_coarse2,
            "delta3": delta3,
            "final": x_final,
            "stage1_sign_logits": stage1_out["sign_logits"],
            "stage1_sign_map": stage1_out["sign_map"],
            "stage1_mag_map": stage1_out["mag_map"],
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


def to_signed_vis(img: np.ndarray, percentile: float = VIS_PERCENTILE):
    if img.ndim == 3:
        img = img.mean(axis=2)

    abs_max = float(np.percentile(np.abs(img), percentile))
    if abs_max < 1e-12:
        abs_max = 1.0

    normalized = np.clip(img / abs_max, -1.0, 1.0)
    colored = plt.get_cmap("seismic")((normalized + 1.0) * 0.5)[..., :3]
    return (colored * 255.0).astype(np.uint8), abs_max


def to_unit_gray_vis(img: np.ndarray):
    if img.ndim == 3:
        img = img.mean(axis=2)
    clipped = np.clip(img, 0.0, 1.0)
    gray = (clipped * 255.0).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def to_image_shifted_vis(img: np.ndarray):
    if img.ndim == 3:
        img = img.mean(axis=2)
    clipped = np.clip(img + 0.5, 0.0, 1.0)
    gray = (clipped * 255.0).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def build_vis(img: np.ndarray, mode: str):
    if mode == "signed":
        vis, abs_max = to_signed_vis(img)
        return vis, f"vis=[{-abs_max:.6f},{abs_max:.6f}]"

    if mode == "around_one":
        vis, abs_max = to_signed_vis(img - 1.0)
        return vis, f"vis=[{1.0 - abs_max:.6f},{1.0 + abs_max:.6f}]"

    if mode == "unit":
        return to_unit_gray_vis(img), "vis=[0,1]"

    if mode == "image_shifted":
        return to_image_shifted_vis(img), "vis=clip(x+0.5,0,1)"

    raise ValueError(f"Unknown vis mode: {mode}")


def ensure_dirs():
    for item_name in OUTPUT_SPECS:
        (OUTPUT_ROOT / item_name).mkdir(parents=True, exist_ok=True)


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    return ckpt


def _adapt_state_dict_for_model(state_dict, model):
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict

    first_key = next(iter(state_dict))
    ckpt_has_module_prefix = first_key.startswith("module.")
    model_has_module_prefix = isinstance(model, nn.DataParallel)

    if ckpt_has_module_prefix == model_has_module_prefix:
        return state_dict

    converted = OrderedDict()
    if ckpt_has_module_prefix and not model_has_module_prefix:
        for key, val in state_dict.items():
            converted[key[len("module."):]] = val
        return converted

    if model_has_module_prefix and not ckpt_has_module_prefix:
        for key, val in state_dict.items():
            converted[f"module.{key}"] = val
        return converted

    return state_dict


def build_model(device: torch.device):
    model = CascadeRefine3Stage(**MODEL_CFG)
    model = nn.DataParallel(model).to(device)

    ckpt = torch.load(str(MODEL_PATH), map_location=device)
    state_dict = _extract_state_dict(ckpt)
    state_dict = _adapt_state_dict_for_model(state_dict, model)
    model.load_state_dict(state_dict, strict=True)
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
    print(f"[Info] Model cfg: {MODEL_CFG}")

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
            for item_name, spec in OUTPUT_SPECS.items():
                arr = tensor_to_numpy_img(outputs[item_name])
                out_dir = OUTPUT_ROOT / item_name
                tiff_path = out_dir / f"{name}_{item_name}.tiff"
                png_path = out_dir / f"{name}_{item_name}_vis.png"

                save_float32_tiff(tiff_path, arr)
                vis, vis_range_desc = build_vis(arr, mode=spec["mode"])
                imageio.imwrite(str(png_path), vis)
                stats.append(
                    f"{item_name}: min={arr.min():.6f}, max={arr.max():.6f}, {vis_range_desc}"
                )

            print(f"[{idx}/{len(common_names)}] {name} | " + " | ".join(stats))

    print("[Done] Exported all stage/intermediate tensors as TIFF + PNG.")


if __name__ == "__main__":
    main()
