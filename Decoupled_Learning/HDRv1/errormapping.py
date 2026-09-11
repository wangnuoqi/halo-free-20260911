from pathlib import Path
from collections import defaultdict
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import os

#### 可视化pred和gt结果，产生一个error mapping 图。可以结合中间图的mask以及最终的error来看 ####
# ===== 路径 =====
# s2r
# GT_DIR = Path("/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/result/detail_x5_s2r_v11/gt_enh")       # GT tiff 所在文件夹
# PRED_DIR = Path("/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/result/detail_x5_s2r_v11/pred_enh")    # 模型 output tiff 所在文件夹
# save_vis_dir = "/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/result/detail_x5_s2r_v11/vis_gt_pred"
# os.makedirs(save_vis_dir, exist_ok=True)

# single hdr
# GT_DIR = Path("/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/result/detail_x5_singlehdr_v11/gt_enh")       # GT tiff 所在文件夹
# PRED_DIR = Path("/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/result/detail_x5_singlehdr_v11/pred_enh")    # 模型 output tiff 所在文件夹
# save_vis_dir = "/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/result/detail_x5_singlehdr_v11/vis_gt_pred"
# os.makedirs(save_vis_dir, exist_ok=True)

# single hdr
GT_DIR = Path("/mnt/d/nuoqi/S2R-HDR/data/test_mixmodel/testoutput_rgb")       # GT tiff 所在文件夹
PRED_DIR = Path("/mnt/d/nuoqi/S2R-HDR/data/test_mixmodel/model_outputrgb")    # 模型 output tiff 所在文件夹
save_vis_dir = "/mnt/d/nuoqi/S2R-HDR/data/test_mixmodel/vis"
os.makedirs(save_vis_dir, exist_ok=True)

def visualize_error_map_only(result, save_dir=None, fixed_vmax=0.1):
    """
    fixed_vmax: 
        用于对比时的固定上限。
        建议设置为 1.0 (代表100%相对误差) 或 0.5 (代表50%)。
        如果设为 None，则退回自适应模式（不推荐用于对比）。
    """
    key     = result["key"]
    psnr    = result["psnr"]
    err_vis = result["err_vis"] # pred结果相对于gt的误差百分比
    
    # ================= 修改重点 =================
    if fixed_vmax is not None:
        # 【对比模式】大家都用同一把尺子
        vmax = fixed_vmax
    else:
        # 【单张分析模式】自适应，只为了看清这一张图内部的分布
        vmax = np.quantile(err_vis, 0.99)
        if vmax <= 1e-5: vmax = 1.0

    # 2) 归一化到 [0,1]
    err_norm = np.clip(err_vis / vmax, 0, 1)

    # 3) 画图
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(err_norm, cmap="jet", vmin=0, vmax=0.1) # 以0.1作为上限去展示abs error。
    ax.set_title(f"{key}  Error Map\nPSNR: {psnr:.2f} dB")
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized error")

    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{key}_rgberrmap.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()

def process_one_pair_np(prefix_key, gt_path: Path, pred_path: Path):
    gt   = read_tiff(gt_path)    # -> [0,1] float32
    pred = read_tiff(pred_path)

    if gt.shape != pred.shape:
        raise ValueError(f"Shape 不一致: key={prefix_key}, gt={gt.shape}, pred={pred.shape}")

    psnr, mse = calc_psnr_np(pred, gt)
    _, err_vis = compute_error_map_np(pred, gt, mode="abs")  # 采用不同的error 方式，包括相对百分比和绝对值差

    return {
        "key": prefix_key,
        "psnr": psnr,
        "mse": mse,
        "err_vis": err_vis,
    }

def compute_error_map_np(pred: np.ndarray, target: np.ndarray, mode="rel"):
    """
    pred / target: numpy 数组，已经归一化到 [0,1]
    mode:
      - "abs": 绝对误差 |pred - target|
      - "sq" : 平方误差 (pred - target)^2
    返回:
      err_full: 和输入同 shape 的误差 (逐通道)
      err_vis : H×W 的单通道误差，用于可视化
    """
    diff = pred - target

    if mode == "abs":
        err_full = np.abs(diff)
    elif mode == "sq":
        err_full = diff ** 2
    elif mode == "rel":
        eps = 1e-9
        err_full = diff / (target + eps)
    else:
        raise ValueError("mode must be 'abs' or 'sq'")

    # 多通道就对通道求平均，得到一个 2D map
    if err_full.ndim == 3:
        err_vis = err_full.mean(axis=2)
    else:
        err_vis = err_full

    return err_full, err_vis

def calc_psnr_np(pred: np.ndarray, target: np.ndarray):
    """
    pred / target: numpy 数组，已归一化到 [0,1]
    """
    diff = pred - target
    mse = np.mean(diff ** 2)
    psnr = 100 if mse == 0 else 10 * np.log10(1.0 / mse)
    return psnr, mse

def get_prefix_key(path: Path):
    """
    从文件名形如 xx_xx_xx.tif 中取第一个前缀作为 key：
    例如 '123_45_67.tif' -> '123'
    """
    stem = path.stem          # 去掉扩展名: '123_45_67'
    first = stem.split('_')[0]
    return first

def read_tiff(path: Path):
    """
    读取 tiff，返回 float32 的 numpy 数组，并尽量归一化到 [0, 1]
    """
    img = tiff.imread(str(path))         # H×W 或 H×W×C
    arr = img.astype(np.float32)

    if img.dtype == np.uint8:
        arr = arr / 255.0
    elif img.dtype == np.uint16:
        arr = arr / 65535.0
    else:
        # 如果是浮点类型，根据最大值简单归一化一下
        max_val = float(arr.max()) if arr.size > 0 else 1.0
        if max_val > 1.0:
            arr = arr / max_val

    return arr

def collect_tiffs(folder: Path):
    exts = [".tif", ".tiff"]
    files = []
    for ext in exts:
        files.extend(folder.glob(f"*{ext}"))
    return sorted(files)

def build_prefix_pairs(gt_dir: Path, pred_dir: Path):
    gt_files   = collect_tiffs(gt_dir)
    pred_files = collect_tiffs(pred_dir)

    # 1) 先按“第一个前缀”分组
    gt_group   = defaultdict(list)  # key -> [gt_path1, gt_path2, ...]
    pred_group = defaultdict(list)  # key -> [pred_path1, pred_path2, ...]

    for p in gt_files:
        k = get_prefix_key(p)
        gt_group[k].append(p)

    for p in pred_files:
        k = get_prefix_key(p)
        pred_group[k].append(p)

    # 2) 找出两个文件夹里都出现的 key
    common_keys = sorted(set(gt_group.keys()) & set(pred_group.keys()))

    print(f"GT 里有 {len(gt_group)} 个不同前缀")
    print(f"Pred 里有 {len(pred_group)} 个不同前缀")
    print(f"两边都存在的前缀有 {len(common_keys)} 个")

    # 3) 按前缀建立配对
    #    如果某个前缀下有多张图，这里用 zip 一一配对（数量不等的话，多出来的会被丢弃）
    pairs = []   # 每个元素: (prefix_key, gt_path, pred_path)
    for k in common_keys:
        gt_list   = sorted(gt_group[k])
        pred_list = sorted(pred_group[k])

        if len(gt_list) != len(pred_list):
            print(f"[WARN] 前缀 {k}：GT 有 {len(gt_list)} 张, Pred 有 {len(pred_list)} 张，将按最小数量配对")

        for gt_p, pr_p in zip(gt_list, pred_list):
            pairs.append((k, gt_p, pr_p))

    print(f"总共得到 {len(pairs)} 对 (GT, Pred) 可用于 PSNR / error mapping")
    return pairs

pairs = build_prefix_pairs(GT_DIR, PRED_DIR) # 找对应前缀文件
all_psnr = []


for prefix_key, gt_path, pred_path in pairs:
    res = process_one_pair_np(prefix_key, gt_path, pred_path)
    all_psnr.append(res["psnr"])
    visualize_error_map_only(res, save_dir=save_vis_dir, fixed_vmax=0.5)

print("样本数:", len(all_psnr))
print("平均 PSNR:", np.mean(all_psnr))
print("PSNR 标准差:", np.std(all_psnr))