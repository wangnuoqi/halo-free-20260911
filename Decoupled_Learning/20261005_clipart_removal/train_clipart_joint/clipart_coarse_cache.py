"""Float32 coarse cache and aligned geometric augmentation."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from clipart_conv_base_common import DeepGuidedFilter, make_dataset, ClipartJpegDataset


def checkpoint_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_augment(items):
    # One shared transform for raw/coarse/GT; no interpolation or recompression.
    rotation = random.randrange(4)
    flip = random.random() < 0.5
    result = []
    for item in items:
        item = np.rot90(item, rotation, axes=(-2, -1))
        if flip:
            item = np.flip(item, axis=-1)
        result.append(np.ascontiguousarray(item))
    return result


class CachedCoarseDataset(Dataset):
    def __init__(self, data_root, quality, split, list_dir, channels, cache_dir, augment=False):
        self.source = make_dataset(data_root, quality, split, list_dir, channels)
        self.channels = channels
        self.augment = augment
        self.cache_dir = Path(cache_dir) / split
        for jpg, _ in self.source.pairs:
            if not (self.cache_dir / (jpg.stem + '.npy')).is_file():
                raise FileNotFoundError(f'Missing coarse cache: {self.cache_dir / jpg.stem}')

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        jpg, png = self.source.pairs[index]
        raw = ClipartJpegDataset._read(jpg, self.channels)
        gt = ClipartJpegDataset._read(png, self.channels)
        coarse = np.load(self.cache_dir / (jpg.stem + '.npy'), allow_pickle=False)
        if coarse.shape != raw.shape or gt.shape != raw.shape or coarse.dtype != np.float32:
            raise ValueError(f'Invalid coarse cache: {jpg.stem}')
        items = [coarse, raw, gt]
        if self.augment:
            items = paired_augment(items)
        return tuple(torch.from_numpy(item) for item in items)


def prepare_cache(data_root, list_dir, quality, channels, base_checkpoint, cache_dir,
                  batch_size=8, num_workers=16, device=None):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata = {'channels': channels, 'quality': quality,
                'base_sha256': checkpoint_hash(base_checkpoint),
                'data_root': str(Path(data_root).resolve()), 'dtype': 'float32',
                'list_sha256': {split: checkpoint_hash(Path(list_dir) / f'{split}_list.txt')
                                for split in ('train', 'val')}}
    manifest = cache_dir / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != metadata:
        raise ValueError('Cache metadata mismatch: choose a new cache directory')
    manifest.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    device = torch.device(device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    base = DeepGuidedFilter(channels)
    saved = torch.load(base_checkpoint, map_location='cpu')
    state = saved.get('model_state_dict', saved)
    base.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
    base.requires_grad_(False).eval().to(device)
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        base = torch.nn.DataParallel(base)
    for split in ('train', 'val'):
        target = cache_dir / split
        target.mkdir(exist_ok=True)
        dataset = make_dataset(data_root, quality, split, list_dir, channels)
        # Resume interrupted caching; atomic rename keeps incomplete files out.
        dataset.pairs = [(jpg, png) for jpg, png in dataset.pairs
                         if not (target / (jpg.stem + '.npy')).exists()]
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                            shuffle=False, pin_memory=device.type == 'cuda')
        offset = 0
        with torch.inference_mode():
            for lr, hr, _ in tqdm(loader, desc=f'Cache C{channels} {split}'):
                outputs = base(lr.to(device)-0.5, hr.to(device)-0.5).cpu().numpy()
                for output, (jpg, _) in zip(outputs, dataset.pairs[offset:offset+len(outputs)]):
                    path = target / (jpg.stem + '.npy')
                    tmp = path.with_suffix('.tmp')
                    with tmp.open('wb') as file:
                        np.save(file, output.astype(np.float32), allow_pickle=False)
                    tmp.replace(path)
                offset += len(outputs)
        print(f'{split}: wrote {offset} new coarse files', flush=True)
    del base
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--channels', type=int, choices=(1, 3), required=True)
    p.add_argument('--quality', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=16)
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    tag = 'a' if args.channels == 1 else 'b'
    prepare_cache(root.parent/'openclipart_5000/jpeg_pairs_512', root.parent/'openclipart_5000',
                  args.quality, args.channels,
                  root/f'ckpt/exp1-{tag}_conv_base/convguided_exp1{tag}_q{args.quality}_best.pth',
                  root.parent/f'openclipart_5000/coarse_cache/exp1-{tag}_q{args.quality}', args.batch_size, args.num_workers)


if __name__ == '__main__':
    main()
