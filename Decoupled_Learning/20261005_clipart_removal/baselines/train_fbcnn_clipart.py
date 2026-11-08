"""Official color FBCNN, retrained on fixed OpenClipart JPEG/clean pairs."""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'FBCNN'))
from models.network_fbcnn import FBCNN
from models.select_network import init_weights


class Pairs(Dataset):
    def __init__(self, data, lists, split, quality, patch_size=128):
        self.train = split == 'train'
        self.patch = patch_size
        self.q = (100-quality)/100.0
        stems = [s.strip() for s in (lists/f'{split}_list.txt').read_text().splitlines() if s.strip()]
        if not stems or len(stems) != len(set(stems)):
            raise ValueError('Empty or duplicated split list')
        self.pairs = [(data/f'jpeg_q{quality:02d}/{s}.jpg', data/f'clean_png/{s}.png') for s in stems]
        for pair in self.pairs:
            for path in pair:
                if not path.is_file():
                    raise FileNotFoundError(path)
        print(f'{split}: {len(self.pairs)} pairs, Q{quality}', flush=True)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        arrays = []
        for path in self.pairs[i]:
            with Image.open(path) as image:
                arrays.append(np.asarray(image.convert('RGB'), dtype=np.float32)/255.0)
        if arrays[0].shape != arrays[1].shape:
            raise ValueError(f'Unaligned pair {self.pairs[i]}')
        if self.train:
            h, w = arrays[0].shape[:2]
            if min(h,w) < self.patch:
                raise ValueError('Image smaller than patch')
            y, x = random.randint(0,h-self.patch), random.randint(0,w-self.patch)
            # Keep the protocol aligned with ARCNN/VIVNet: random paired crop
            # only, with no geometric or color augmentation.
            arrays = [a[y:y+self.patch, x:x+self.patch] for a in arrays]
        tensors = [torch.from_numpy(np.ascontiguousarray(a.transpose(2,0,1))) for a in arrays]
        return tensors[0], tensors[1], torch.tensor([self.q], dtype=torch.float32)


def main():
    p = argparse.ArgumentParser()
    project = ROOT.parent
    p.add_argument('--data-root', type=Path, default=project/'openclipart_5000/jpeg_pairs_512')
    p.add_argument('--list-dir', type=Path, default=project/'openclipart_5000')
    p.add_argument('--output-dir', type=Path, default=ROOT/'ckpt/fbcnn_rgb_q30_protocol160')
    p.add_argument('--quality', type=int, default=30)
    p.add_argument('--epochs', type=int, default=160)
    p.add_argument('--batch-size', type=int, default=8, help='total batch across GPUs')
    p.add_argument('--num-workers', type=int, default=16)
    p.add_argument('--patch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--val-every', type=int, default=5, help='full validation interval in epochs; final epoch is always validated')
    p.add_argument('--resume', type=Path)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if a.val_every < 1:
        p.error('--val-every must be >= 1')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    model = FBCNN(in_nc=3, out_nc=3, nc=[64,128,256,512], nb=4,
                  act_mode='BR', downsample_mode='strideconv', upsample_mode='convtranspose')
    saved = torch.load(a.resume, map_location='cpu') if a.resume else None
    if saved:
        if saved['config']['quality'] != a.quality or saved['config']['patch_size'] != a.patch_size:
            raise ValueError('Resume quality/patch size mismatch')
        model.load_state_dict(saved['model_state_dict'], strict=True)
    else:
        init_weights(model, init_type='orthogonal', init_bn_type='uniform', gain=0.2)
    model.to(device)
    raw = model
    gpus = torch.cuda.device_count() if device.type == 'cuda' else 0
    if gpus > 1:
        if a.batch_size < gpus:
            p.error('Batch size must be at least GPU count')
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    # Official milestones are optimizer iterations, NOT epochs.  Compress the
    # schedule to the local 160-epoch budget: 40/80/120 epochs.
    # The train loader uses drop_last=True, so match its complete-batch count.
    steps_per_epoch = math.floor(3722 / a.batch_size)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[40 * steps_per_epoch, 80 * steps_per_epoch, 120 * steps_per_epoch],
        gamma=0.5)
    start, step, best = 0, 0, float('inf')
    if saved:
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        scheduler.load_state_dict(saved['scheduler_state_dict'])
        start, step, best = saved['epoch'], saved['global_step'], saved['best_val_mse']
    loaders = {split: DataLoader(Pairs(a.data_root,a.list_dir,split,a.quality,a.patch_size),
        batch_size=a.batch_size if split=='train' else 1, shuffle=split=='train',
        num_workers=a.num_workers, pin_memory=device.type=='cuda',
        persistent_workers=a.num_workers > 0, drop_last=split=='train') for split in ('train','val')}
    if not len(loaders['train']):
        raise ValueError('No complete training batches')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    (a.output_dir/'config_used.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(f'FBCNN from scratch: GPUs={gpus}, total batch={a.batch_size}, patch={a.patch_size}; workers={a.num_workers}', flush=True)
    for epoch in range(start,a.epochs):
        model.train(); total, count = 0., 0
        bar = tqdm(loaders['train'],desc=f'FBCNN train {epoch+1}/{a.epochs}')
        for x,gt,q in bar:
            x,gt,q = x.to(device,non_blocking=True),gt.to(device,non_blocking=True),q.to(device,non_blocking=True)
            pred,q_est = model(x)
            image_loss = torch.nn.functional.l1_loss(pred,gt)
            q_loss = torch.nn.functional.l1_loss(q_est,q)
            loss = image_loss + .001*q_loss
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();scheduler.step();step+=1
            total+=loss.item()*x.shape[0];count+=x.shape[0]
            bar.set_postfix(loss=f'{loss.item():.5f}',q_loss=f'{q_loss.item():.4f}',lr=f'{optimizer.param_groups[0]["lr"]:.1e}')
        should_validate = (epoch + 1) % a.val_every == 0 or epoch + 1 == a.epochs
        mse = None; improved = False; psnrs = []; q_error = 0.
        if should_validate:
            model.eval(); sse = elements = 0
            with torch.inference_mode():
                for x,gt,q in tqdm(loaders['val'],desc='FBCNN val full images'):
                    x,gt,q=x.to(device),gt.to(device),q.to(device)
                    pred,q_est=model(x); diff=(pred.clamp(0,1)-gt).square()
                    image_mse=diff.mean().item(); sse+=diff.sum().item(); elements+=gt.numel()
                    psnrs.append(-10*math.log10(max(image_mse,1e-12)))
                    q_error+=torch.nn.functional.l1_loss(q_est,q).item()
            mse=sse/elements; improved=mse<best;best=min(best,mse)
        state={'epoch':epoch+1,'global_step':step,'best_val_mse':best,
               'model_state_dict':raw.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
               'scheduler_state_dict':scheduler.state_dict(),'config':config}
        torch.save(state,a.output_dir/'fbcnn_last.pth')
        if improved:
            torch.save(state,a.output_dir/'fbcnn_best.pth')
        record={'epoch':epoch+1,'global_step':step,'train_loss':total/count,'val_mse':mse,
                'val_psnr_global':-10*math.log10(max(mse,1e-12)) if mse is not None else None,
                'val_psnr_mean':sum(psnrs)/len(psnrs) if psnrs else None,
                'val_q_mae':q_error/len(psnrs) if psnrs else None}
        with (a.output_dir/'history.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(record)+'\n')
        print(record,flush=True)


if __name__ == '__main__':
    main()
