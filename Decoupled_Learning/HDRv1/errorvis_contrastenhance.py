#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import numpy as np
import imageio.v3 as iio  # pip install imageio
from typing import Optional, Dict
# ----------------- 你的路径 -----------------
# s2r data
INPUT_DIR = Path("/mnt/d/nuoqi/S2R-HDR/data/pipelinev1/input_test_log")  # p (log), 0900.tiff
GT_DIR    = Path("/mnt/d/nuoqi/S2R-HDR/data/pipelinev1/testlabel_log")   # q_gt (log), 0900.tiff
PRED_DIR  = Path("/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/result/rf_output_s2r_v11")  # q_pred (log), 0900_pred_pred.tiff

# single hdr data
# INPUT_DIR = Path("/mnt/d/nuoqi/S2R-HDR/data/Single-Image-HDR/testinput")  # p (log), 0900.tiff
# GT_DIR    = Path("/mnt/d/nuoqi/S2R-HDR/data/Single-Image-HDR/testlabel")   # q_gt (log), 0900.tiff
# PRED_DIR  = Path("/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/result/rf_output_s2r_v11")  # q_pred (log), 0900_pred_pred.tiff

OUT_DIR = Path("/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/result/detail_x5_s2r_v11")
#OUT_DIR = Path("/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/result/base_%5_singlehdr")
FACTOR = 5.0

SUPPORTED_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".hdr", ".exr", ".npy", ".npz"}


def as_float_hwc(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:
        x = x[..., None]
    # heuristic: CHW -> HWC
    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
        x = np.transpose(x, (1, 2, 0))
    return x.astype(np.float32, copy=False)


def load_any(path: Path) -> np.ndarray:
    suf = path.suffix.lower()
    if suf == ".npy":
        return as_float_hwc(np.load(path))
    if suf == ".npz":
        z = np.load(path)
        key = "arr_0" if "arr_0" in z.files else z.files[0]
        return as_float_hwc(z[key])
    return as_float_hwc(iio.imread(path))


def save_tiff_float32(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.asarray(arr, dtype=np.float32))


def list_imgs(folder: Path):
    return sorted([p for p in folder.rglob("*")
                   if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])


def leading_digits_str(filename: str) -> Optional[str]:
    """取文件名开头的数字串：0900.tiff -> '0900' ; 0900_pred_pred.tiff -> '0900'"""
    m = re.match(r"^(\d+)", filename)
    return m.group(1) if m else None


def build_id_index(folder: Path) -> dict[str, Path]:
    """用“开头数字串”作为 key 建索引"""
    idx: dict[str, Path] = {}
    for p in list_imgs(folder):
        k = leading_digits_str(p.name)
        if k is not None:
            idx[k] = p
    return idx


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pred = OUT_DIR / "pred_enh"
    out_gt   = OUT_DIR / "gt_enh"
    out_pred.mkdir(parents=True, exist_ok=True)
    out_gt.mkdir(parents=True, exist_ok=True)

    inputs = list_imgs(INPUT_DIR)
    if not inputs:
        raise RuntimeError(f"No input images found in: {INPUT_DIR}")

    pred_idx = build_id_index(PRED_DIR)  # key: '0900' -> 0900_pred_pred.tiff
    gt_idx = build_id_index(GT_DIR)  # gt同名 0900.tiff

    miss_pred = 0
    miss_gt = 0

    for p_path in inputs:
        # 提取 Input ID: 0001.tiff -> 0001
        k = leading_digits_str(p_path.name) 
        
        # 3. 统一用 ID 查字典
        pred_path = pred_idx.get(k, None)
        gt_path = gt_idx.get(k, None) # 这样 '0001' 就能查到那个复杂的 GT 文件了

        if pred_path is None:
            miss_pred += 1
            continue
        if gt_path is None:
            miss_gt += 1
            continue

        p_log = load_any(p_path)
        q_pred = load_any(pred_path)
        q_gt = load_any(gt_path)

        if p_log.shape != q_pred.shape:
            raise ValueError(f"[{p_path.name}] shape mismatch: p {p_log.shape} vs pred {q_pred.shape} ({pred_path.name})")
        if p_log.shape != q_gt.shape:
            raise ValueError(f"[{p_path.name}] shape mismatch: p {p_log.shape} vs gt {q_gt.shape} ({gt_path.name})")

        # enh = q + 5*(p-q)  detail * 5
        enh_pred = q_pred + FACTOR * (p_log - q_pred)
        enh_gt   = q_gt   + FACTOR * (p_log - q_gt)
        # base / 5
        # enh_pred = (q_pred / FACTOR) + (p_log - q_pred)
        # enh_gt   = (q_gt   / FACTOR) + (p_log - q_gt)

        # 输出用input原名 0900.tiff
        save_tiff_float32(out_pred / f"{p_path.stem}_pred{p_path.suffix}", enh_pred)
        save_tiff_float32(out_gt   / p_path.name, enh_gt)

        print(f"[OK] {p_path.name}  pred={pred_path.name}")

    print(f"Done. Missing matches: pred={miss_pred}, gt={miss_gt}")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
