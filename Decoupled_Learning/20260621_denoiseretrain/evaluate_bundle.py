"""Evaluate one new Exp7/Exp8 bundle under the shared BSD68 protocol."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).parent; sys.path.insert(0, str(ROOT))
from exp7_8_common import BSDSOnline, Exp6Refine, Exp7CRefine, add_official_test_noise, collate_pad, denoise_forward, load_denoiser


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--bundle', required=True); parser.add_argument('--test-root', required=True); parser.add_argument('--out', required=True); parser.add_argument('--device', default='cuda:0')
    args=parser.parse_args(); bundle_path=Path(args.bundle); bundle=torch.load(bundle_path, map_location='cpu', weights_only=False); metrics=bundle['metrics']
    device=torch.device(args.device); denoiser=load_denoiser(metrics['denoiser'], metrics['checkpoint'], device); denoiser.load_state_dict(bundle['denoiser'], strict=True)
    refine=(Exp7CRefine() if metrics['rf'] == 'exp7c' else Exp6Refine('exp6d')).to(device); refine.load_state_dict(bundle['refine'], strict=True); denoiser.eval(); refine.eval()
    loader=DataLoader(BSDSOnline(args.test_root, 'full', train=False), batch_size=1, shuffle=False, collate_fn=collate_pad)
    coarse_scores=[]; final_scores=[]
    with torch.no_grad():
        for batch in loader:
            clean=batch['clean'].to(device); mask=batch['mask'].to(device); noisy=add_official_test_noise(clean)
            coarse=denoise_forward(denoiser, noisy, metrics['denoiser']).clamp(0, 1); final=refine(noisy, coarse)['final'].clamp(0, 1)
            for image, target in ((coarse, coarse_scores), (final, final_scores)):
                mse=((image-clean).square()*mask).sum()/mask.sum(); target.append(float(10*torch.log10(1/mse)))
    record={'bundle':str(bundle_path), 'source_epoch':bundle['epoch'], 'denoiser':metrics['denoiser'], 'protocol':metrics['protocol'], 'mode':metrics['mode'], 'rf':metrics['rf'], 'coarse_psnr':sum(coarse_scores)/len(coarse_scores), 'final_psnr':sum(final_scores)/len(final_scores)}
    output=Path(args.out); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(record, indent=2)); print(json.dumps(record, indent=2))


if __name__ == '__main__': main()
