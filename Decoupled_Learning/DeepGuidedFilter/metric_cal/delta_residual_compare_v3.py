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
GT_DIR = Path("/mnt/d/nuoqi/S2R-HDR/data/test_vivo/testlabel")
MODEL_PATH = Path(
    "/mnt/d/nuoqi/halo-free/Decoupled_Learning/DeepGuidedFilter/models_retrain/rf_3stage_best_otsu_v3_abrev.pth"
)
OUTPUT_ROOT = Path(
    "/mnt/d/nuoqi/halo-free/Decoupled_Learning/DeepGuidedFilter/metric_cal/delta_compare_v3_abrev"
)

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
MODEL_CFG = {
    "base_channels": 32,
    "a_scale": 0.2,
    "b_scale": 0.4,
    "use_delta_as_input": True,
}

# delta 相关可视化默认使用全数据集 delta_gt 的绝对最大值作为统一色阶。
MANUAL_DELTA_SCALE = None
MANUAL_ERROR_SCALE = None
EPS = 1e-12
SIGN_COMPARE_TAU = 3e-3


class Stage1DeltaNet(nn.Module):
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
        return self.net(inp)


class ABAdjustNet(nn.Module):
    def __init__(self, base_channels=32, a_scale=0.2, b_scale=0.4, use_delta_as_input=True):
        super().__init__()
        self.a_scale = a_scale
        self.b_scale = b_scale
        self.use_delta_as_input = use_delta_as_input

        in_ch = 3 + (1 if use_delta_as_input else 0)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
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
        a = (1.0 + self.a_scale) * torch.tanh(a_raw)
        b = self.b_scale * torch.tanh(b_raw)
        return a, b


class CascadeRefine3Stage(nn.Module):
    def __init__(self, base_channels=32, a_scale=0.2, b_scale=0.4, use_delta_as_input=True):
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


def crop_to_common(*imgs):
    min_h = min(img.shape[0] for img in imgs)
    min_w = min(img.shape[1] for img in imgs)
    return tuple(img[:min_h, :min_w] for img in imgs)


def crop_to_even(tensor: torch.Tensor):
    _, _, h, w = tensor.shape
    nh = h if h % 2 == 0 else h - 1
    nw = w if w % 2 == 0 else w - 1
    return tensor[:, :, :nh, :nw]


def tensor_to_numpy_2d(x: torch.Tensor):
    arr = x.detach().cpu().numpy().astype(np.float32)
    return arr[0, 0]


def save_float32_tiff(path: Path, img: np.ndarray):
    imageio.imwrite(str(path), img.astype(np.float32))


def to_signed_vis_fixed(img: np.ndarray, abs_max: float):
    if abs_max <= 0:
        abs_max = 1.0
    normalized = np.clip(img / abs_max, -1.0, 1.0)
    colored = plt.get_cmap("seismic")((normalized + 1.0) * 0.5)[..., :3]
    return (colored * 255.0).astype(np.uint8)


def compute_sign_metrics(pred: np.ndarray, gt: np.ndarray, tau: float):
    valid = np.abs(gt) > tau
    valid_count = int(valid.sum())
    if valid_count == 0:
        return 1.0, 0.0, 0.0, valid_count

    pred_pos = pred > tau
    pred_neg = pred < -tau
    gt_pos = gt > tau
    gt_neg = gt < -tau

    correct = (pred_pos & gt_pos) | (pred_neg & gt_neg)
    wrong = (pred_pos & gt_neg) | (pred_neg & gt_pos)
    uncertain = valid & ~(correct | wrong)

    acc = float(correct[valid].mean())
    wrong_rate = float(wrong[valid].mean())
    uncertain_rate = float(uncertain[valid].mean())
    return acc, wrong_rate, uncertain_rate, valid_count


def build_sign_compare_vis(pred: np.ndarray, gt: np.ndarray, tau: float):
    valid = np.abs(gt) > tau

    pred_pos = pred > tau
    pred_neg = pred < -tau
    gt_pos = gt > tau
    gt_neg = gt < -tau

    correct = valid & ((pred_pos & gt_pos) | (pred_neg & gt_neg))
    wrong = valid & ((pred_pos & gt_neg) | (pred_neg & gt_pos))
    uncertain = valid & ~(correct | wrong)

    vis = np.zeros((*gt.shape, 3), dtype=np.uint8)
    vis[...] = np.array([128, 128, 128], dtype=np.uint8)
    vis[correct] = np.array([60, 200, 60], dtype=np.uint8)
    vis[wrong] = np.array([230, 70, 70], dtype=np.uint8)
    vis[uncertain] = np.array([240, 200, 70], dtype=np.uint8)
    return vis


def ensure_dirs():
    for name in (
        "delta_gt",
        "delta_pred",
        "stage1_pred",
        "stage2_pred",
        "stage3_pred",
        "delta_error",
        "stage1_sign_cmp",
        "stage2_sign_cmp",
        "stage3_sign_cmp",
        "final_pred",
        "summary",
    ):
        (OUTPUT_ROOT / name).mkdir(parents=True, exist_ok=True)


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
        for k, v in state_dict.items():
            converted[k[len("module."):]] = v
        return converted

    if model_has_module_prefix and not ckpt_has_module_prefix:
        for k, v in state_dict.items():
            converted[f"module.{k}"] = v
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


def write_csv(path: Path, rows, header):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")


def main():
    for p, n in (
        (ORIGIN_INPUT_DIR, "ORIGIN_INPUT_DIR"),
        (COARSE_INPUT_DIR, "COARSE_INPUT_DIR"),
        (GT_DIR, "GT_DIR"),
    ):
        if not p.is_dir():
            raise FileNotFoundError(f"{n} not found: {p}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")

    ensure_dirs()
    origin_map = collect_images(ORIGIN_INPUT_DIR)
    coarse_map = collect_images(COARSE_INPUT_DIR)
    gt_map = collect_images(GT_DIR)
    common = sorted(set(origin_map) & set(coarse_map) & set(gt_map))
    if not common:
        raise RuntimeError("No matched filenames among origin/coarse/gt.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)

    print(f"[Info] Matched triplets: {len(common)}")
    print(f"[Info] Device: {device}")
    print(f"[Info] Output root: {OUTPUT_ROOT}")

    records = []
    global_delta_absmax = 0.0
    global_error_absmax = 0.0

    with torch.no_grad():
        for idx, name in enumerate(common, start=1):
            origin = read_image(origin_map[name])
            coarse = read_image(coarse_map[name])
            gt = read_image(gt_map[name])
            origin, coarse, gt = crop_to_common(origin, coarse, gt)

            x_raw = torch.from_numpy(origin).unsqueeze(0).unsqueeze(0)
            x_coarse0 = torch.from_numpy(coarse).unsqueeze(0).unsqueeze(0)
            x_gt = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)

            x_raw = crop_to_even(x_raw).to(device) - 0.5
            x_coarse0 = crop_to_even(x_coarse0).to(device) - 0.5
            x_gt = crop_to_even(x_gt).to(device) - 0.5

            out = model(x_raw, x_coarse0)
            delta1_pred = tensor_to_numpy_2d(out["delta1"])
            delta2_pred = tensor_to_numpy_2d(out["delta2"])
            delta3_pred = tensor_to_numpy_2d(out["delta3"])
            delta_pred = delta3_pred
            delta_gt = tensor_to_numpy_2d(x_gt - x_coarse0)
            delta_err = delta_pred - delta_gt
            final_pred = tensor_to_numpy_2d(out["final"] + 0.5)

            mae = float(np.mean(np.abs(delta_err)))
            rmse = float(np.sqrt(np.mean(delta_err * delta_err)))
            maxae = float(np.max(np.abs(delta_err)))
            gt_energy = float(np.mean(np.abs(delta_gt)))
            nmae = float(mae / max(gt_energy, EPS))

            s1_acc, s1_wrong, s1_uncertain, sign_valid_cnt = compute_sign_metrics(
                delta1_pred, delta_gt, SIGN_COMPARE_TAU
            )
            s2_acc, s2_wrong, s2_uncertain, _ = compute_sign_metrics(
                delta2_pred, delta_gt, SIGN_COMPARE_TAU
            )
            s3_acc, s3_wrong, s3_uncertain, _ = compute_sign_metrics(
                delta3_pred, delta_gt, SIGN_COMPARE_TAU
            )

            records.append(
                (
                    name,
                    mae,
                    rmse,
                    maxae,
                    nmae,
                    gt_energy,
                    s1_acc,
                    s2_acc,
                    s3_acc,
                    s1_wrong,
                    s2_wrong,
                    s3_wrong,
                    s1_uncertain,
                    s2_uncertain,
                    s3_uncertain,
                    sign_valid_cnt,
                )
            )
            global_delta_absmax = max(global_delta_absmax, float(np.max(np.abs(delta_gt))))
            global_error_absmax = max(global_error_absmax, float(np.max(np.abs(delta_err))))

            save_float32_tiff(OUTPUT_ROOT / "delta_gt" / f"{name}_delta_gt.tiff", delta_gt)
            save_float32_tiff(OUTPUT_ROOT / "delta_pred" / f"{name}_delta_pred.tiff", delta_pred)
            save_float32_tiff(OUTPUT_ROOT / "stage1_pred" / f"{name}_stage1_pred.tiff", delta1_pred)
            save_float32_tiff(OUTPUT_ROOT / "stage2_pred" / f"{name}_stage2_pred.tiff", delta2_pred)
            save_float32_tiff(OUTPUT_ROOT / "stage3_pred" / f"{name}_stage3_pred.tiff", delta3_pred)
            save_float32_tiff(OUTPUT_ROOT / "delta_error" / f"{name}_delta_error.tiff", delta_err)
            save_float32_tiff(OUTPUT_ROOT / "final_pred" / f"{name}_final_pred.tiff", final_pred)

            print(
                f"[Pass1 {idx}/{len(common)}] {name} | "
                f"MAE={mae:.6f} RMSE={rmse:.6f} MaxAE={maxae:.6f} nMAE={nmae:.6f} | "
                f"SignAcc(s1/s2/s3)={s1_acc:.4f}/{s2_acc:.4f}/{s3_acc:.4f} "
                f"(valid={sign_valid_cnt})"
            )

    delta_scale = float(MANUAL_DELTA_SCALE) if MANUAL_DELTA_SCALE is not None else global_delta_absmax
    error_scale = float(MANUAL_ERROR_SCALE) if MANUAL_ERROR_SCALE is not None else global_error_absmax
    delta_scale = max(delta_scale, 1e-6)
    error_scale = max(error_scale, 1e-6)

    print(f"[Info] Unified delta scale from delta_gt absmax: +/-{delta_scale:.6f}")
    print(f"[Info] Unified error scale: +/-{error_scale:.6f}")

    for idx, record in enumerate(records, start=1):
        (
            name,
            mae,
            rmse,
            maxae,
            nmae,
            gt_energy,
            s1_acc,
            s2_acc,
            s3_acc,
            s1_wrong,
            s2_wrong,
            s3_wrong,
            s1_uncertain,
            s2_uncertain,
            s3_uncertain,
            sign_valid_cnt,
        ) = record

        delta_gt = read_image(OUTPUT_ROOT / "delta_gt" / f"{name}_delta_gt.tiff")
        delta_pred = read_image(OUTPUT_ROOT / "delta_pred" / f"{name}_delta_pred.tiff")
        stage1_pred = read_image(OUTPUT_ROOT / "stage1_pred" / f"{name}_stage1_pred.tiff")
        stage2_pred = read_image(OUTPUT_ROOT / "stage2_pred" / f"{name}_stage2_pred.tiff")
        stage3_pred = read_image(OUTPUT_ROOT / "stage3_pred" / f"{name}_stage3_pred.tiff")
        delta_err = read_image(OUTPUT_ROOT / "delta_error" / f"{name}_delta_error.tiff")
        final_pred = read_image(OUTPUT_ROOT / "final_pred" / f"{name}_final_pred.tiff")

        imageio.imwrite(
            OUTPUT_ROOT / "delta_gt" / f"{name}_delta_gt_vis_fixed.png",
            to_signed_vis_fixed(delta_gt, delta_scale),
        )
        imageio.imwrite(
            OUTPUT_ROOT / "delta_pred" / f"{name}_delta_pred_vis_fixed.png",
            to_signed_vis_fixed(delta_pred, delta_scale),
        )
        imageio.imwrite(
            OUTPUT_ROOT / "stage1_pred" / f"{name}_stage1_pred_vis_fixed.png",
            to_signed_vis_fixed(stage1_pred, delta_scale),
        )
        imageio.imwrite(
            OUTPUT_ROOT / "stage2_pred" / f"{name}_stage2_pred_vis_fixed.png",
            to_signed_vis_fixed(stage2_pred, delta_scale),
        )
        imageio.imwrite(
            OUTPUT_ROOT / "stage3_pred" / f"{name}_stage3_pred_vis_fixed.png",
            to_signed_vis_fixed(stage3_pred, delta_scale),
        )
        imageio.imwrite(
            OUTPUT_ROOT / "delta_error" / f"{name}_delta_error_vis_fixed.png",
            to_signed_vis_fixed(delta_err, error_scale),
        )
        imageio.imwrite(
            OUTPUT_ROOT / "stage1_sign_cmp" / f"{name}_stage1_sign_cmp.png",
            build_sign_compare_vis(stage1_pred, delta_gt, SIGN_COMPARE_TAU),
        )
        imageio.imwrite(
            OUTPUT_ROOT / "stage2_sign_cmp" / f"{name}_stage2_sign_cmp.png",
            build_sign_compare_vis(stage2_pred, delta_gt, SIGN_COMPARE_TAU),
        )
        imageio.imwrite(
            OUTPUT_ROOT / "stage3_sign_cmp" / f"{name}_stage3_sign_cmp.png",
            build_sign_compare_vis(stage3_pred, delta_gt, SIGN_COMPARE_TAU),
        )
        imageio.imwrite(
            OUTPUT_ROOT / "final_pred" / f"{name}_final_pred_gray.png",
            (np.clip(final_pred, 0.0, 1.0) * 255.0).astype(np.uint8),
        )

        print(
            f"[Pass2 {idx}/{len(records)}] {name} | "
            f"saved fixed-scale pngs + sign-compare maps | MAE={mae:.6f} RMSE={rmse:.6f} | "
            f"SignAcc(s1/s2/s3)={s1_acc:.4f}/{s2_acc:.4f}/{s3_acc:.4f}"
        )

    mean_mae = float(np.mean([r[1] for r in records]))
    mean_rmse = float(np.mean([r[2] for r in records]))
    mean_maxae = float(np.mean([r[3] for r in records]))
    mean_nmae = float(np.mean([r[4] for r in records]))
    mean_s1_acc = float(np.mean([r[6] for r in records]))
    mean_s2_acc = float(np.mean([r[7] for r in records]))
    mean_s3_acc = float(np.mean([r[8] for r in records]))
    mean_s1_wrong = float(np.mean([r[9] for r in records]))
    mean_s2_wrong = float(np.mean([r[10] for r in records]))
    mean_s3_wrong = float(np.mean([r[11] for r in records]))

    write_csv(
        OUTPUT_ROOT / "summary" / "residual_metrics.csv",
        records,
        header=[
            "name",
            "mae",
            "rmse",
            "maxae",
            "nmae",
            "gt_abs_mean",
            "sign_acc_stage1",
            "sign_acc_stage2",
            "sign_acc_stage3",
            "sign_wrong_stage1",
            "sign_wrong_stage2",
            "sign_wrong_stage3",
            "sign_uncertain_stage1",
            "sign_uncertain_stage2",
            "sign_uncertain_stage3",
            "sign_valid_count",
        ],
    )
    write_csv(
        OUTPUT_ROOT / "summary" / "global_scale.csv",
        [(delta_scale, error_scale)],
        header=["delta_scale_absmax", "error_scale_absmax"],
    )
    write_csv(
        OUTPUT_ROOT / "summary" / "dataset_mean_metrics.csv",
        [
            (
                mean_mae,
                mean_rmse,
                mean_maxae,
                mean_nmae,
                mean_s1_acc,
                mean_s2_acc,
                mean_s3_acc,
                mean_s1_wrong,
                mean_s2_wrong,
                mean_s3_wrong,
            )
        ],
        header=[
            "mean_mae",
            "mean_rmse",
            "mean_maxae",
            "mean_nmae",
            "mean_sign_acc_stage1",
            "mean_sign_acc_stage2",
            "mean_sign_acc_stage3",
            "mean_sign_wrong_stage1",
            "mean_sign_wrong_stage2",
            "mean_sign_wrong_stage3",
        ],
    )

    print("[Done] Residual comparison finished.")
    print(
        f"[Done] Dataset mean | MAE={mean_mae:.6f} RMSE={mean_rmse:.6f} "
        f"MaxAE={mean_maxae:.6f} nMAE={mean_nmae:.6f} | "
        f"SignAcc(s1/s2/s3)={mean_s1_acc:.4f}/{mean_s2_acc:.4f}/{mean_s3_acc:.4f}"
    )


if __name__ == "__main__":
    main()
