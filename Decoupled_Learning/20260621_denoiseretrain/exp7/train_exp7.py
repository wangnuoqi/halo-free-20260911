"""Train one self-contained Exp7 or Exp8 denoiser-plus-RF run."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parents[1]))
from exp7_8_common import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--denoiser', choices=['convguided', 'dncnn', 'ffdnet'], required=True)
    parser.add_argument('--mode', choices=['joint', 'frozen'], required=True)
    parser.add_argument('--protocol', choices=['full', 'convguided_patch', 'dncnn_official', 'ffdnet_official'], required=True)
    parser.add_argument('--rf', choices=['exp6d', 'exp7c'], required=True)
    parser.add_argument('--train-root', required=True)
    parser.add_argument('--test-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--patch-size', type=int, default=128)
    parser.add_argument('--patches-per-image', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--denoiser-lr', type=float, default=1e-4)
    parser.add_argument('--denoiser-weight-decay', type=float, default=0.0)
    parser.add_argument('--refine-lr', type=float, default=3e-3)
    parser.add_argument('--coarse-loss', choices=['smooth', 'mse', 'mse_smooth'], default='smooth')
    return parser.parse_args()


def train_dataset(args, epoch):
    if args.protocol == 'dncnn_official':
        return OfficialPatchDataset(args.train_root, 'dncnn', epoch=epoch)
    if args.protocol == 'ffdnet_official':
        # FFDNet's source code regenerates its patch set every ten epochs.
        return OfficialPatchDataset(args.train_root, 'ffdnet', epoch=(epoch // 10) * 10)
    if args.protocol == 'convguided_patch':
        return BSDSOnline(args.train_root, 'patch', args.patch_size, args.patches_per_image, True, style='convguided')
    return BSDSOnline(args.train_root, 'full', train=True, style=args.denoiser)


def loader(dataset, batch_size, workers, shuffle=True, drop_last=True):
    kwargs = {'batch_size': batch_size, 'shuffle': shuffle, 'drop_last': drop_last, 'num_workers': workers,
              'pin_memory': True, 'collate_fn': collate_pad}
    if workers:
        kwargs['persistent_workers'] = False
    return DataLoader(dataset, **kwargs)


def apply_lrs(optimizer, epoch, protocol, mode, refine_lr, denoiser_lr):
    if mode == 'frozen':
        den_scale = rf_scale = .5 ** (epoch // 60)
    elif protocol == 'dncnn_official':
        den_scale = 1.0 if epoch < 30 else .1
        rf_scale = 1.0 if epoch < 30 else .5
    elif protocol == 'ffdnet_official':
        den_scale = 3.0 ** -(epoch // 100)
        rf_scale = den_scale
    else:
        den_scale = rf_scale = .5 ** (epoch // 60)
    for group in optimizer.param_groups:
        group['lr'] = (refine_lr * rf_scale) if group['name'] == 'refine' else (denoiser_lr * den_scale)


def main():
    args = parse_args()
    if (args.mode == 'joint') != (args.rf == 'exp7c'):
        raise ValueError('joint must use exp7c; frozen must use exp6d')
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    torch.manual_seed(0); np.random.seed(0)
    denoiser = load_denoiser(args.denoiser, args.checkpoint, device)
    refine = (Exp7CRefine() if args.rf == 'exp7c' else Exp6Refine('exp6d')).to(device)
    if args.mode == 'frozen':
        denoiser.eval()
        for parameter in denoiser.parameters(): parameter.requires_grad_(False)
    else:
        for parameter in denoiser.parameters(): parameter.requires_grad_(True)
    gpu_count = torch.cuda.device_count() if device.type == 'cuda' else 0
    if gpu_count > 1:
        denoiser = torch.nn.DataParallel(denoiser)
        refine = torch.nn.DataParallel(refine)
    groups = [{'name': 'refine', 'params': refine.parameters(), 'lr': args.refine_lr}]
    if args.mode == 'joint':
        groups.append({'name': 'denoiser', 'params': denoiser.parameters(), 'lr': args.denoiser_lr, 'weight_decay': args.denoiser_weight_decay})
    optimizer = torch.optim.Adam(groups)
    test_loader = loader(BSDSOnline(args.test_root, 'full', train=False), max(1, gpu_count), 0, shuffle=False, drop_last=False)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    manifest = vars(args) | {'gpu_count': gpu_count, 'noise_train': 'numpy_random_sigma25_no_clamp',
                             'noise_test': 'numpy_randomstate0_per_image_no_clamp'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    best = -float('inf')
    label = f'{args.denoiser}_{args.protocol}_{args.mode}_{args.rf}'
    trainable_parameters = list(refine.parameters())
    if args.mode == 'joint':
        trainable_parameters += list(denoiser.parameters())
    for epoch in range(args.epochs):
        apply_lrs(optimizer, epoch, args.protocol, args.mode, args.refine_lr, args.denoiser_lr)
        refine.train(); denoiser.train(args.mode == 'joint')
        dataset = train_dataset(args, epoch)
        # Patch records stay grouped by source image so each worker reuses its image cache.
        grouped_patches = isinstance(dataset, OfficialPatchDataset) or dataset.protocol == 'patch'
        train_loader = loader(dataset, args.batch_size, args.num_workers, shuffle=not grouped_patches)
        total_loss = torch.zeros((), device=device)
        train_progress = tqdm(train_loader, desc=f'{label} {epoch + 1}/{args.epochs}',
                              unit='batch', dynamic_ncols=True, leave=False)
        for batch_index, batch in enumerate(train_progress, start=1):
            clean = batch['clean'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)
            noisy = add_official_train_noise(clean)
            coarse = denoise_forward(denoiser, noisy, args.denoiser).clamp(0, 1)
            outputs = refine(noisy, coarse)
            rf_loss, _ = exp7c_loss(outputs, coarse, clean, mask) if args.rf == 'exp7c' else exp6_loss(outputs, coarse, clean, mask)
            coarse_term = coarse_loss(coarse, clean, mask, args.coarse_loss)
            loss = rf_loss + coarse_term if args.mode == 'joint' else rf_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 5.0)
            optimizer.step()
            total_loss.add_(loss.detach())
            if batch_index % 100 == 0 or batch_index == len(train_loader):
                train_progress.set_postfix(loss=f'{loss.detach().item():.5f}', lr=f'{optimizer.param_groups[0]["lr"]:.1e}')
        denoiser.eval(); refine.eval(); coarse_scores=[]; final_scores=[]
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f'{label} eval {epoch + 1}/{args.epochs}',
                              unit='image', dynamic_ncols=True, leave=False):
                clean=batch['clean'].to(device); mask=batch['mask'].to(device)
                noisy=add_official_test_noise(clean)
                coarse=denoise_forward(denoiser, noisy, args.denoiser).clamp(0, 1)
                final=refine(noisy, coarse)['final'].clamp(0, 1)
                coarse_scores.extend(masked_psnr_per_image(coarse, clean, mask).cpu().tolist())
                final_scores.extend(masked_psnr_per_image(final, clean, mask).cpu().tolist())
        coarse_psnr=sum(coarse_scores)/len(coarse_scores); final_psnr=sum(final_scores)/len(final_scores)
        mean_loss=(total_loss/max(1,len(train_loader))).item()
        print(f'[{label}] {epoch+1}/{args.epochs} loss={mean_loss:.6f} coarse={coarse_psnr:.4f} final={final_psnr:.4f}', flush=True)
        if final_psnr > best:
            best=final_psnr
            metrics=manifest | {'epoch': epoch+1, 'coarse_psnr': coarse_psnr, 'final_psnr': final_psnr}
            save_bundle(output / f'{label}_best.pth', denoiser, refine, metrics, epoch+1)
    (output / 'training_complete.json').write_text(json.dumps({'epochs': args.epochs, 'best_final_psnr': best}, indent=2))


if __name__ == '__main__':
    main()
