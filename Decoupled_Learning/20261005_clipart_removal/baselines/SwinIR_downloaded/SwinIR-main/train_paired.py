import argparse, json, random, sys, time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paired_data import PairedJpegDataset, full_loader
from models.network_swinir import SwinIR

def main():
    p = argparse.ArgumentParser(); p.add_argument('--data-root', required=True); p.add_argument('--list-dir', required=True); p.add_argument('--quality', type=int, default=30); p.add_argument('--output-dir', required=True)
    p.add_argument('--epochs', type=int, default=50); p.add_argument('--batch-size', type=int, default=8); p.add_argument('--patch-size', type=int, default=126); p.add_argument('--lr', type=float, default=2e-4); p.add_argument('--val-every', type=int, default=10, help='Run full validation every N epochs and always on the final epoch.'); p.add_argument('--workers', type=int, default=16); p.add_argument('--seed', type=int, default=42); p.add_argument('--resume', type=str, default=None)
    p.add_argument('--lr-milestones', type=int, nargs='*', default=[], help='Epochs after which to halve the learning rate.')
    p.add_argument('--micro-batch-size', type=int, default=None,
                   help='Split each optimizer batch into smaller gradient-accumulation chunks.')
    args = p.parse_args()
    if args.patch_size < 1: p.error('--patch-size must be positive')
    if args.micro_batch_size is not None and not 1 <= args.micro_batch_size <= args.batch_size:
        p.error('--micro-batch-size must be between 1 and --batch-size')
    if args.val_every < 1: p.error('--val-every must be >= 1')
    if any(x < 1 or x >= args.epochs for x in args.lr_milestones) or args.lr_milestones != sorted(set(args.lr_milestones)):
        p.error('--lr-milestones must be unique, increasing epochs before --epochs')
    random.seed(args.seed); torch.manual_seed(args.seed); out = Path(args.output_dir)
    if not args.resume and (out / 'last.pth').exists():
        p.error(f'{out / "last.pth"} already exists; pass --resume to continue')
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; refusing to start this GPU training run on CPU')
    device = torch.device('cuda')
    print(f'[init] device={device}, visible_gpus={torch.cuda.device_count()}, workers={args.workers}', flush=True)
    print('[init] Building SwinIR and moving it to the device...', flush=True)
    window_size = 7
    model_size = ((args.patch_size + window_size - 1) // window_size) * window_size
    model = SwinIR(img_size=model_size, patch_size=1, in_chans=3, embed_dim=180, depths=[6,6,6,6,6,6], num_heads=[6,6,6,6,6,6], window_size=window_size, mlp_ratio=2, img_range=255., upscale=1, upsampler='', resi_connection='1conv').to(device)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    print('[init] Model ready. Checking training pairs...', flush=True)
    train = DataLoader(PairedJpegDataset(args.data_root, args.list_dir, args.quality, 'train', 3, args.patch_size), args.batch_size, True, num_workers=args.workers, pin_memory=True)
    print('[init] Training pairs ready. Checking validation pairs...', flush=True)
    val = DataLoader(full_loader(args.data_root, args.list_dir, args.quality, 'val', 3), 1, False, num_workers=0)
    print(f'[data] train={len(train.dataset)}, val={len(val.dataset)}, batches_per_epoch={len(train)}, total_batch_size={args.batch_size}', flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=args.lr_milestones, gamma=0.5)
    best = float('inf'); history = []
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        previous = ckpt.get('config', {})
        for key in ('quality', 'patch_size', 'batch_size', 'lr', 'lr_milestones'):
            if key in previous and previous[key] != getattr(args, key):
                raise ValueError(f'Resume {key} mismatch: {previous[key]} != {getattr(args, key)}')
        raw = model.module if isinstance(model, nn.DataParallel) else model
        raw.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt: opt.load_state_dict(ckpt['optimizer'])
        if args.lr_milestones and 'scheduler' not in ckpt:
            raise ValueError('Cannot resume milestone schedule from a checkpoint without scheduler state')
        if 'scheduler' in ckpt: scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        best = float(ckpt.get('best_val_mse', float('inf')))
        history_path = out / 'history.json'
        if history_path.exists():
            history = [row for row in json.loads(history_path.read_text()) if row['epoch'] < start_epoch]
        if history and history[-1]['epoch'] != start_epoch - 1:
            raise ValueError('History and checkpoint epoch do not match')
        if best == float('inf') and history:
            best = min((row['val_mse'] for row in history if row['val_mse'] is not None), default=best)
        print(f'[resume] loaded {args.resume}; continuing at epoch {start_epoch}', flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); total = 0.0
        started = time.perf_counter()
        print(f'[train] epoch {epoch}/{args.epochs}: waiting for first batch...', flush=True)
        for step, (low, high) in enumerate(train, 1):
            if step == 1: print(f'[train] First batch loaded: {tuple(low.shape)}. Running forward/backward...', flush=True)
            low, high = low.to(device, non_blocking=True), high.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            batch_loss = 0.0
            micro_size = args.micro_batch_size or low.shape[0]
            for offset in range(0, low.shape[0], micro_size):
                chunk_low = low[offset:offset + micro_size]
                chunk_high = high[offset:offset + micro_size]
                pred = model(chunk_low)
                loss = nn.functional.l1_loss(pred, chunk_high)
                weight = chunk_low.shape[0] / low.shape[0]
                (loss * weight).backward()
                batch_loss += loss.item() * weight
            opt.step(); total += batch_loss
            if step == 1 or step % 10 == 0 or step == len(train):
                print(f'[train] epoch={epoch}/{args.epochs} batch={step}/{len(train)} l1={batch_loss:.6f} elapsed={time.perf_counter() - started:.1f}s', flush=True)
        should_validate = epoch % args.val_every == 0 or epoch == args.epochs
        val_mse = None
        if should_validate:
            model.eval(); val_mse = 0.0
            print(f'[val] epoch {epoch}: starting {len(val)} full images...', flush=True)
            with torch.no_grad():
                for step, (low, high) in enumerate(val, 1):
                    val_mse += nn.functional.mse_loss(model(low.to(device)), high.to(device)).item()
                    if step == 1 or step % 25 == 0 or step == len(val):
                        print(f'[val] image={step}/{len(val)} mean_mse={val_mse / step:.8f}', flush=True)
            val_mse /= len(val)
        row = {'epoch': epoch, 'train_l1': total / len(train), 'val_mse': val_mse,
               'lr': opt.param_groups[0]['lr']}; history.append(row); print(row, flush=True)
        improved = val_mse is not None and val_mse < best
        if improved: best = val_mse
        scheduler.step()
        raw = model.module if isinstance(model, nn.DataParallel) else model
        state = {'epoch': epoch, 'model': raw.state_dict(), 'optimizer': opt.state_dict(),
                 'scheduler': scheduler.state_dict(), 'best_val_mse': best, 'config': vars(args)}
        torch.save(state, out / 'last.pth')
        if improved: torch.save(state, out / 'best.pth')
        (out / 'history.json').write_text(json.dumps(history, indent=2))
if __name__ == '__main__': main()
