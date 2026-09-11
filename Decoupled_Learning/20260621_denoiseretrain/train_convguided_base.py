"""Train an experiment-local sigma25 base denoiser before RF attachment."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from exp7_8_common import BSDSOnline, add_official_test_noise, add_official_train_noise, collate_pad, denoise_forward, load_denoiser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--denoiser', choices=['convguided', 'dncnn', 'ffdnet'], required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--train-root', required=True); parser.add_argument('--test-root', required=True)
    parser.add_argument('--out', required=True); parser.add_argument('--protocol', choices=['full', 'patch'], required=True)
    parser.add_argument('--epochs', type=int, required=True); parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--patch-size', type=int, default=128); parser.add_argument('--patches-per-image', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4); parser.add_argument('--num-workers', type=int, default=8); parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.manual_seed(0); np.random.seed(0); device = torch.device(args.device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    if args.denoiser != 'convguided' and not args.checkpoint:
        parser.error('--checkpoint is required for DnCNN and FFDNet base fine-tuning')
    model = load_denoiser(args.denoiser, args.checkpoint, device)
    # The third-party adapters default to frozen inference models; base training
    # must explicitly restore gradients for every denoiser implementation.
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)
    train_set = BSDSOnline(args.train_root, args.protocol, args.patch_size, args.patches_per_image, True, style=args.denoiser)
    test_set = BSDSOnline(args.test_root, 'full', train=False)
    kwargs={'num_workers': args.num_workers, 'pin_memory': True, 'collate_fn': collate_pad}
    train_loader=DataLoader(train_set, batch_size=args.batch_size, shuffle=args.protocol != 'patch', drop_last=True, **kwargs)
    test_loader=DataLoader(test_set, batch_size=max(1, torch.cuda.device_count()), shuffle=False, collate_fn=collate_pad)
    optimizer=torch.optim.Adam(model.parameters(), lr=args.lr)
    out=Path(args.out); out.mkdir(parents=True, exist_ok=False); best=-float('inf')
    manifest=vars(args) | {'model':args.denoiser, 'loss':'masked_mse', 'noise_train':'numpy_random_sigma25_no_clamp', 'noise_test':'numpy_randomstate0_per_image_no_clamp'}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    for epoch in range(args.epochs):
        for group in optimizer.param_groups:
            group['lr'] = args.lr * (.5 ** (epoch // 60))
        model.train(); total=torch.zeros((), device=device)
        train_progress = tqdm(train_loader, desc=f'{args.denoiser}-{args.protocol} {epoch + 1}/{args.epochs}',
                              unit='batch', dynamic_ncols=True, leave=False)
        for batch_index, batch in enumerate(train_progress, start=1):
            clean=batch['clean'].to(device, non_blocking=True); mask=batch['mask'].to(device, non_blocking=True)
            output=denoise_forward(model, add_official_train_noise(clean), args.denoiser)
            loss=((output-clean).square()*mask).sum()/mask.sum()
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step(); total.add_(loss.detach())
            if batch_index % 100 == 0 or batch_index == len(train_loader):
                train_progress.set_postfix(loss=f'{loss.detach().item():.5f}', lr=f'{optimizer.param_groups[0]["lr"]:.1e}')
        model.eval(); scores=[]
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f'{args.denoiser} eval {epoch + 1}/{args.epochs}',
                              unit='image', dynamic_ncols=True, leave=False):
                clean=batch['clean'].to(device); mask=batch['mask'].to(device)
                output=denoise_forward(model, add_official_test_noise(clean), args.denoiser).clamp(0, 1)
                mse=((output-clean).square()*mask).sum((1,2,3))/mask.sum((1,2,3)).clamp_min(1.0); scores.extend((10*torch.log10(1/mse)).cpu().tolist())
        score=sum(scores)/len(scores); mean_loss=(total/max(1,len(train_loader))).item(); print(f'[{args.denoiser}-{args.protocol}] {epoch+1}/{args.epochs} loss={mean_loss:.6f} psnr={score:.4f}', flush=True)
        if score > best:
            best=score
            state=model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save({'model_state_dict':state, 'metrics':manifest | {'epoch':epoch+1, 'psnr':score}}, out/f'{args.denoiser}_best.pth')
    (out / 'training_complete.json').write_text(json.dumps({'epochs': args.epochs, 'best_psnr': best}, indent=2))


if __name__ == '__main__': main()
