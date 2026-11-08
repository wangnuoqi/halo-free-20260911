"""Frozen clip-art base + independently trained three-stage refine."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from clipart_conv_base_common import DeepGuidedFilter, make_dataset, seed_everything
from clipart_refine_models import CascadeRefine3Stage, cascade_loss
from clipart_coarse_cache import checkpoint_hash, CachedCoarseDataset


def main(channels=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--channels', type=int, choices=(1, 3), default=channels or 1)
    parser.add_argument('--quality', type=int, default=30)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--list-dir', type=Path)
    parser.add_argument('--base-checkpoint', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=8, help='total batch across all GPUs')
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--cached-coarse', action='store_true', help='load existing float32 cache; never run base')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--augment', action='store_true', help='aligned flips/90-degree rotations in cached training')
    args = parser.parse_args()
    if channels is not None and args.channels != channels:
        parser.error('This entry point has a fixed channel count')
    root = Path(__file__).resolve().parent
    project = root.parent
    tag = 'a' if args.channels == 1 else 'b'
    args.data_root = args.data_root or project / 'openclipart_5000/jpeg_pairs_512'
    args.list_dir = args.list_dir or project / 'openclipart_5000'
    args.base_checkpoint = args.base_checkpoint or root / f'ckpt/exp1-{tag}_conv_base/convguided_exp1{tag}_q{args.quality}_best.pth'
    args.output_dir = args.output_dir or root / f'ckpt/exp2-{tag}_refine'
    args.cache_dir = args.cache_dir or project / f'openclipart_5000/coarse_cache/exp1-{tag}_q{args.quality}'
    if args.augment and not args.cached_coarse:
        parser.error('--augment requires --cached-coarse')
    if args.cached_coarse:
        manifest = args.cache_dir / 'manifest.json'
        metadata = json.loads(manifest.read_text(encoding='utf-8'))
        expected = {'channels': args.channels, 'quality': args.quality,
                    'base_sha256': checkpoint_hash(args.base_checkpoint),
                    'data_root': str(args.data_root.resolve()), 'dtype': 'float32',
                    'list_sha256': {split: checkpoint_hash(args.list_dir / f'{split}_list.txt')
                                    for split in ('train', 'val')}}
        if metadata != expected:
            raise ValueError('Cache metadata mismatch: select the matching cache/data/base')
    seed_everything(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    base = None
    if not args.cached_coarse:
        base = DeepGuidedFilter(args.channels)
        checkpoint = torch.load(args.base_checkpoint, map_location='cpu')
        state = checkpoint.get('model_state_dict', checkpoint)
        base.load_state_dict({k.removeprefix('module.'): v for k,v in state.items()}, strict=True)
        base.requires_grad_(False)
        base.eval()
    refine = CascadeRefine3Stage(channels=args.channels)
    refine = refine.to(device)
    if base is not None:
        base = base.to(device)
    gpu_count = torch.cuda.device_count() if device.type == 'cuda' else 0
    if gpu_count > 1:
        if args.batch_size < gpu_count:
            parser.error('total batch-size must be at least the visible GPU count')
        if base is not None:
            base = torch.nn.DataParallel(base)
        refine = torch.nn.DataParallel(refine)
    optimizer = torch.optim.Adam(refine.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    start, best = 0, float('inf')
    raw_refine = refine.module if isinstance(refine, torch.nn.DataParallel) else refine
    if args.resume:
        saved = torch.load(args.resume, map_location=device)
        if saved['base_checkpoint'] != str(args.base_checkpoint.resolve()):
            raise ValueError('Resume checkpoint uses a different base')
        raw_refine.load_state_dict(saved['model_state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        scheduler.load_state_dict(saved['scheduler_state_dict'])
        start, best = saved['epoch'], saved['best_loss']
    loaders = {}
    for split in ('train', 'val'):
        if args.cached_coarse:
            ds = CachedCoarseDataset(args.data_root, args.quality, split, args.list_dir,
                                     args.channels, args.cache_dir, augment=args.augment and split=='train')
        else:
            ds = make_dataset(args.data_root, args.quality, split, args.list_dir, args.channels)
        loaders[split] = DataLoader(ds, batch_size=args.batch_size, shuffle=split=='train',
                                   num_workers=args.num_workers, pin_memory=device.type=='cuda',
                                   persistent_workers=args.num_workers > 0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Frozen base: {args.base_checkpoint}; channels={args.channels}; GPUs={gpu_count}; total batch={args.batch_size}', flush=True)
    history = args.output_dir / 'history.jsonl'
    for epoch in range(start, args.epochs):
        refine.train()
        train_loss, n = 0.0, 0
        bar = tqdm(loaders['train'], desc=f'exp2-{tag} train {epoch+1}/{args.epochs}')
        for lr, hr, gt in bar:
            lr, hr, gt = lr.to(device), hr.to(device), gt.to(device)
            with torch.no_grad():
                coarse = lr if args.cached_coarse else base(lr-0.5, hr-0.5)
            outputs = refine(hr-0.5, coarse-0.5)
            loss, _ = cascade_loss(outputs, coarse-0.5, gt-0.5, thresh_mode='otsu', dilate=0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * hr.shape[0]
            n += hr.shape[0]
            bar.set_postfix(loss=f'{loss.item():.5f}', lr=f'{optimizer.param_groups[0]["lr"]:.1e}')
        refine.eval()
        squared, base_squared, count = 0.0, 0.0, 0
        with torch.no_grad():
            for lr, hr, gt in tqdm(loaders['val'], desc=f'exp2-{tag} val {epoch+1}/{args.epochs}'):
                lr, hr, gt = lr.to(device), hr.to(device), gt.to(device)
                coarse = lr if args.cached_coarse else base(lr-0.5, hr-0.5)
                output = (refine(hr-0.5, coarse-0.5)['final']+0.5).clamp(0,1)
                squared += (output-gt).square().sum().item()
                base_squared += (coarse-gt).square().sum().item()
                count += gt.numel()
        val_mse = squared/count
        improved = val_mse < best
        best = min(best, val_mse)
        scheduler.step()
        saved = {'epoch': epoch+1, 'model_state_dict': raw_refine.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                 'best_loss': best, 'channels': args.channels, 'quality': args.quality,
                 'base_checkpoint': str(args.base_checkpoint.resolve()),
                 'cached_coarse': args.cached_coarse, 'augment': args.augment}
        torch.save(saved, args.output_dir / f'refine_exp2{tag}_last.pth')
        if improved:
            torch.save(saved, args.output_dir / f'refine_exp2{tag}_best.pth')
        record = {'epoch': epoch+1, 'train_loss': train_loss/n, 'val_mse': val_mse,
                  'base_val_mse': base_squared/count}
        with history.open('a', encoding='utf-8') as file:
            file.write(json.dumps(record)+'\n')
        print(record, flush=True)


if __name__ == '__main__':
    main()
