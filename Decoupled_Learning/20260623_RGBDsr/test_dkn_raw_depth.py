"""DKN x8 test with raw millimeter depth input (diagnostic only)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'third_party' / 'DKN'))
from models import DKN

class RawDKNTest(Dataset):
    def __init__(self, root):
        root = Path(root)
        self.depth = np.load(root/'test_depth.npy', mmap_mode='r')
        self.rgb = np.load(root/'test_images_v2.npy', mmap_mode='r')
    def __len__(self): return len(self.depth)
    def __getitem__(self, i):
        depth = np.asarray(self.depth[i], np.float32)
        target = depth.copy()
        mn, mx = float(depth.min()), float(depth.max())
        rgb = np.asarray(self.rgb[i], np.float32).transpose(2,0,1) / 255.0
        if self.normalize:
            depth = (depth - depth.min()) / max(depth.max() - depth.min(), 1e-6)
        h,w = depth.shape
        lr = np.asarray(Image.fromarray(depth).resize((w//8,h//8), Image.Resampling.BICUBIC), np.float32)
        up = np.asarray(Image.fromarray(lr).resize((w,h), Image.Resampling.BICUBIC), np.float32)
        return torch.from_numpy(up[None]), torch.from_numpy(np.ascontiguousarray(rgb)), torch.from_numpy(target[None]), mn, mx

def main(a):
    device = torch.device(a.device)
    model = DKN(3, 15, residual=True).to(device)
    state = torch.load(a.checkpoint, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state: state = state['state_dict']
    if all(k.startswith('module.') for k in state): state = {k[7:]:v for k,v in state.items()}
    model.load_state_dict(state, strict=True); model.eval()
    RawDKNTest.normalize = bool(a.normalize)
    ds = RawDKNTest(a.data); n = min(len(ds), a.limit) if a.limit else len(ds)
    rows=[]
    with torch.inference_mode():
        for i in range(n):
            depth, rgb, target, mn, mx = ds[i]
            depth, rgb, target = depth.unsqueeze(0).to(device), rgb.unsqueeze(0).to(device), target.unsqueeze(0).to(device)
            pred = model((rgb, depth))[0,0]
            if a.normalize:
                pred = pred * (mx - mn) + mn
            diff = (pred[6:-6,6:-6] - target[0,0,6:-6,6:-6])
            rows.append(float(diff.square().mean().sqrt().cpu()))
            print(json.dumps({'image': i, 'rmse_mm': rows[-1]}), flush=True)
    out = {'method':'dkn','input':'per_image_minmax_01' if a.normalize else 'raw_millimeters','scale':8,'crop':6,'images':n,
           'rmse_mm':float(np.mean(rows)),'rmse_cm':float(np.mean(rows)/10),
           'checkpoint':str(a.checkpoint)}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--device',default='cpu'); p.add_argument('--limit',type=int,default=0); p.add_argument('--normalize',action='store_true'); main(p.parse_args())
