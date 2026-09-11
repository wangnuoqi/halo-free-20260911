"""Run exp7B RF inference on the test_fei TIFF triplets."""
from __future__ import annotations
import argparse, importlib.util, math
from pathlib import Path
import cv2
import numpy as np
import torch

EDGE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = Path('/mnt/d/nuoqi/S2R-HDR/data/test_fei')

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_model_definitions(path, name):
    """Load class definitions without executing the training bootstrap."""
    source = path.read_text(encoding='utf-8')
    marker = "\ndevice = torch.device('cuda:0')"
    if marker not in source:
        return load_module(path, name)
    namespace = {'__name__': name, '__file__': str(path)}
    exec(compile(source.split(marker, 1)[0], str(path), 'exec'), namespace)
    return type(name, (), namespace)

def load_state(model, path):
    state = torch.load(path, map_location='cpu')
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    if all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)

def even(t):
    h, w = t.shape[-2:]
    return t[..., :h - h % 2, :w - w % 2]


def match_test_triplets(data_root):
    """Match full-resolution input, downsampled input, and label by filename stem."""
    data_root = Path(data_root)
    def scene_key(path):
        stem = path.stem.split('-WLS-', 1)[0]
        return stem[:-len('-log')] if stem.endswith('-log') else stem

    downsampled = {scene_key(p): p for p in (data_root / 'testinput_downsample').iterdir() if p.is_file()}
    labels = {}
    for path in (data_root / 'testlabel').iterdir():
        if path.is_file():
            labels.setdefault(scene_key(path), path)
    triplets = []
    for origin in sorted((data_root / 'testinput').iterdir()):
        if not origin.is_file():
            continue
        key = scene_key(origin)
        if key not in downsampled or key not in labels:
            raise FileNotFoundError(f'{origin.stem}: missing filename-matched test triplet')
        triplets.append((origin, downsampled[key], labels[key]))
    if not triplets:
        raise RuntimeError(f'No test triplets found under {data_root}')
    return triplets


def format_metrics(mse, psnr):
    return f'MSE: {mse:.8f}\nPSNR: {psnr:.4f} dB\n'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    rf_source = load_model_definitions(EDGE_ROOT / 'experiments/exp7B_joint_mixed.py', 'rf_test_fei')
    conv_source = rf_source
    conv = conv_source.DeepGuidedFilter(radius=1, eps=1e-8).to(device)
    rf = rf_source.CascadeRefine3Stage(base_channels=32, a_scale=0.2, b_scale=0.2, use_delta_as_input=True).to(device)
    model_dir = EDGE_ROOT / 'experiments/exp7/exp7B_mixed_joint/models/exp7B_mixed_joint'
    load_state(conv, model_dir / 'convguided_joint_best_20260818_l0p500_a1p400.pth')
    load_state(rf, model_dir / 'rf_3stage_best_20260818_l0p500_a1p400.pth')
    conv.eval(); rf.eval()
    out = args.output_root; out.mkdir(parents=True, exist_ok=True)
    values = []
    with torch.inference_mode():
        for inp, lr_path, label_path in match_test_triplets(args.data_root):
            scene = inp.stem
            origin_np = cv2.imread(str(inp), cv2.IMREAD_UNCHANGED).astype('float32')
            lr_np = cv2.imread(str(lr_path), cv2.IMREAD_UNCHANGED).astype('float32')
            label_np = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED).astype('float32')
            origin = even(torch.from_numpy(origin_np)[None, None].to(device))
            label = even(torch.from_numpy(label_np)[None, None].to(device))
            lr = torch.from_numpy(lr_np)[None, None].to(device)
            coarse = conv(lr - 0.5, origin - 0.5)
            final = torch.clamp(rf(origin - 0.5, coarse - 0.5)['final'] + 0.5, 0, 1)
            pred = final[0, 0].cpu().numpy().astype('float32')
            mse = float(np.mean((pred - label[0, 0].cpu().numpy()) ** 2)); values.append(mse)
            cv2.imwrite(str(out / f'{scene}-rf.png'), np.clip(pred * 255, 0, 255).astype('uint8'))
            print(f'{scene}: MSE={mse:.8f}')
    mean = sum(values) / len(values); psnr = 100 if mean == 0 else 10 * math.log10(1 / mean)
    report = format_metrics(mean, psnr)
    (out / 'metrics.txt').write_text(report, encoding='utf-8'); print(report)

if __name__ == '__main__':
    main()
