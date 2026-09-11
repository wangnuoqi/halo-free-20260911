"""Online denoiser + edge-preserving refine experiments.

This module is intentionally self-contained under 20260621_denoiseretrain.
It does not modify or import the original edgepreserving experiment scripts.
"""
from __future__ import annotations
import json, sys
from collections import OrderedDict
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

SIGMA = 25.0
_LANCZOS_WEIGHT_CACHE = {}

def gray_files(root):
    return sorted(p for p in Path(root).iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp'})

def read_gray(path):
    x = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if x is None: raise FileNotFoundError(path)
    h, w = x.shape
    x = x[:h-h%2, :w-w%2].astype(np.float32) / 255.0
    return torch.from_numpy(x[None])

class BSDSOnline(Dataset):
    def __init__(self, root, protocol='full', patch_size=64, patches_per_image=8, train=True, seed=0, style='generic'):
        self.files = gray_files(root); self.protocol=protocol; self.patch_size=patch_size
        self.train=train; self.seed=seed; self.style=style
        self.mult = patches_per_image if protocol == 'patch' and train else 1
        self._image_cache = OrderedDict()
    def __len__(self): return len(self.files) * self.mult
    def __getitem__(self, i):
        file_index = i // self.mult if self.protocol == 'patch' and self.train else i % len(self.files)
        path=self.files[file_index]; clean=self._clean_image(file_index)
        if self.protocol == 'patch':
            ps=self.patch_size; h,w=clean.shape[-2:]
            if self.train and self.style == 'convguided':
                mode=int(torch.randint(0,8,(1,)))
                if mode & 1: clean=torch.flip(clean,[-2])
                if mode & 2: clean=torch.flip(clean,[-1])
                if mode & 4: clean=clean.transpose(-2,-1)
            elif self.train and self.style == 'dncnn':
                scale=(1.0, 0.9, 0.8, 0.7)[i % 4]
                clean=F.interpolate(clean[None], scale_factor=scale, mode='bicubic', align_corners=False).clamp(0,1)[0]
                mode=int(torch.randint(0,8,(1,)))
                if mode & 1: clean=torch.flip(clean,[-2])
                if mode & 2: clean=torch.flip(clean,[-1])
                if mode & 4: clean=clean.transpose(-2,-1)
            elif self.train and self.style == 'ffdnet':
                # Official FFDNet generatepatches.m: 0.45 + 0.05*randi(21).
                scale=(45 + 5 * int(torch.randint(0,21,(1,)))) / 100.0
                clean=F.interpolate(clean[None], scale_factor=scale, mode='bicubic', align_corners=False).clamp(0,1)[0]
                mode=int(torch.randint(0,8,(1,)))
                if mode & 1: clean=torch.flip(clean,[-2])
                if mode & 2: clean=torch.flip(clean,[-1])
                if mode & 4: clean=clean.transpose(-2,-1)
            h,w=clean.shape[-2:]
            if h < ps or w < ps: clean=F.pad(clean,(0,max(0,ps-w),0,max(0,ps-h)),mode='reflect')
            h,w=clean.shape[-2:]
            if self.train:
                top=int(torch.randint(0,h-ps+1,(1,))); left=int(torch.randint(0,w-ps+1,(1,)))
            else:
                top=(h-ps)//2; left=(w-ps)//2
            clean=clean[...,top:top+ps,left:left+ps]
        elif self.train:
            mode=int(torch.randint(0,8,(1,)))
            if mode & 1: clean=torch.flip(clean,[-2])
            if mode & 2: clean=torch.flip(clean,[-1])
            if mode & 4: clean=clean.transpose(-2,-1)
        return {'clean':clean, 'name':path.name, 'seed':self.seed+i*1009}

    def _clean_image(self, file_index):
        cached = self._image_cache.pop(file_index, None)
        if cached is None:
            cached = read_gray(self.files[file_index])
        self._image_cache[file_index] = cached
        if len(self._image_cache) > 16:
            self._image_cache.popitem(last=False)
        return cached


class OfficialPatchDataset(Dataset):
    """Dense-grid patch protocol used by the original DnCNN and FFDNet code."""
    def __init__(self, root, denoiser, seed=0, epoch=0):
        if denoiser not in {'dncnn', 'ffdnet'}:
            raise ValueError(f'OfficialPatchDataset does not support {denoiser}')
        self.files = gray_files(root)
        self.denoiser = denoiser
        self.seed = int(seed)
        self._resized_cache = OrderedDict()
        self.rebuild(epoch)

    def rebuild(self, epoch):
        self.epoch = int(epoch)
        patch_size, stride = (40, 10) if self.denoiser == 'dncnn' else (70, 60)
        rng = np.random.default_rng(self.seed + self.epoch)
        records = []
        for file_index in rng.permutation(len(self.files)):
            path = self.files[int(file_index)]
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(path)
            scale_indices = range(4) if self.denoiser == 'dncnn' else (int(rng.integers(0, 21)),)
            augmentation = -1 if self.denoiser == 'dncnn' else int(rng.integers(0, 8))
            for scale_index in scale_indices:
                scale = (1.0, .9, .8, .7)[scale_index] if self.denoiser == 'dncnn' else (45 + 5 * scale_index) / 100.0
                height, width = int(image.shape[0] * scale), int(image.shape[1] * scale)
                for top in range(0, height - patch_size + 1, stride):
                    for left in range(0, width - patch_size + 1, stride):
                        records.append((file_index, scale_index, top, left, augmentation))
        if not records:
            raise RuntimeError(f'No {patch_size}x{patch_size} patches found in {self.files}')
        self.records = np.asarray(records, dtype=np.int32)
        self.scale_indices = self.records[:, 1].copy()

    def __len__(self):
        return len(self.records)

    def _resized_image(self, file_index, scale_index):
        key = (int(file_index), int(scale_index))
        cached = self._resized_cache.pop(key, None)
        if cached is not None:
            self._resized_cache[key] = cached
            return cached
        image = cv2.imread(str(self.files[key[0]]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(self.files[key[0]])
        scale = (1.0, .9, .8, .7)[key[1]] if self.denoiser == 'dncnn' else (45 + 5 * key[1]) / 100.0
        resized = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)), interpolation=cv2.INTER_CUBIC)
        self._resized_cache[key] = resized
        if len(self._resized_cache) > 16:
            self._resized_cache.popitem(last=False)
        return resized

    def __getitem__(self, index):
        file_index, scale_index, top, left, augmentation = self.records[index]
        resized = self._resized_image(file_index, scale_index)
        patch_size = 40 if self.denoiser == 'dncnn' else 70
        patch = resized[int(top):int(top)+patch_size, int(left):int(left)+patch_size]
        mode = int(torch.randint(0, 8, (1,))) if augmentation < 0 else int(augmentation)
        if mode & 1: patch = np.flipud(patch)
        if mode & 2: patch = np.fliplr(patch)
        if mode & 4: patch = patch.T
        return {'clean': torch.from_numpy(np.ascontiguousarray(patch[None])).float().div_(255.0), 'name': self.files[int(file_index)].name}

def collate_pad(batch):
    h=max(x['clean'].shape[-2] for x in batch); w=max(x['clean'].shape[-1] for x in batch)
    ys=[]; mask=[]
    for x in batch:
        y=x['clean']; ph=h-y.shape[-2]; pw=w-y.shape[-1]
        ys.append(F.pad(y,(0,pw,0,ph),mode='reflect' if ph or pw else 'constant'))
        m=torch.zeros(1,h,w); m[...,:y.shape[-2],:y.shape[-1]]=1; mask.append(m)
    return {'clean':torch.stack(ys), 'mask':torch.stack(mask), 'names':[x['name'] for x in batch]}

def add_official_test_noise(clean, sigma=SIGMA):
    out=[]
    for x in clean:
        rng=np.random.RandomState(0)
        noise=rng.normal(0.0, sigma/255.0, tuple(x.shape)).astype(np.float32)
        out.append(x+torch.from_numpy(noise).to(device=x.device, dtype=x.dtype))
    return torch.stack(out)

def add_official_train_noise(clean, sigma=SIGMA):
    # Preserve NumPy's sigma25 samples while avoiding a large float64 batch temporary.
    noisy = clean.clone()
    for image in noisy:
        noise = np.random.normal(0.0, sigma / 255.0, tuple(image.shape)).astype(np.float32)
        image.add_(torch.from_numpy(noise).to(device=image.device, dtype=image.dtype))
    return noisy

def lanczos_downsample_half(x, a=4):
    height, width = x.shape[-2:]
    output_height, output_width = height // 2, width // 2
    if output_height < 1 or output_width < 1:
        raise ValueError(f'Lanczos half-downsample requires dimensions >= 2, got {height}x{width}')

    def weights(length, output_length):
        key = (x.device.type, x.device.index, x.dtype, length, output_length, a)
        if key in _LANCZOS_WEIGHT_CACHE:
            return _LANCZOS_WEIGHT_CACHE[key]
        centers = (torch.arange(output_length, device=x.device, dtype=x.dtype) + .5) * 2 - .5
        base = torch.floor(centers).long() - (a - 1)
        offsets = torch.arange(2 * a, device=x.device).long()
        positions = base[:, None] + offsets[None]
        distance = centers[:, None] - positions.to(x.dtype)
        kernel = torch.sinc(distance) * torch.sinc(distance / a)
        kernel = kernel / kernel.sum(dim=1, keepdim=True).clamp_min(torch.finfo(x.dtype).eps)
        _LANCZOS_WEIGHT_CACHE[key] = (positions, kernel)
        return positions, kernel

    def separable_resize(input_tensor, dimension, output_length):
        axis = dimension % input_tensor.ndim
        length = input_tensor.shape[axis]
        indices, kernel = weights(length, output_length)
        padding = a
        pad_mode = 'reflect' if input_tensor.shape[axis] > padding else 'replicate'
        if axis == input_tensor.ndim - 1:
            padded = F.pad(input_tensor, (padding, padding, 0, 0), mode=pad_mode)
        elif axis == input_tensor.ndim - 2:
            padded = F.pad(input_tensor, (0, 0, padding, padding), mode=pad_mode)
        else:
            raise ValueError(f'Lanczos resize only supports spatial dimensions, got axis {dimension}')
        gather_index = (indices + padding).reshape(-1)
        gather_shape = list(padded.shape)
        gather_shape[axis] = gather_index.numel()
        gather_index = gather_index.view(*([1] * axis), -1, *([1] * (input_tensor.ndim - axis - 1)))
        gather_index = gather_index.expand(*gather_shape)
        sampled = torch.gather(padded, axis, gather_index)
        if axis == input_tensor.ndim - 1:
            sampled = sampled.reshape(*padded.shape[:-1], output_length, 2 * a)
            kernel = kernel.reshape(1, 1, 1, output_length, 2 * a)
            return (sampled * kernel).sum(dim=-1)
        sampled = sampled.reshape(*padded.shape[:-2], output_length, 2 * a, padded.shape[-1])
        kernel = kernel.reshape(1, 1, output_length, 2 * a, 1)
        return (sampled * kernel).sum(dim=-2)

    return separable_resize(separable_resize(x, -1, output_width), -2, output_height)

def denoise_forward(model, noisy, kind, sigma=SIGMA):
    if kind == 'convguided':
        lr=lanczos_downsample_half(noisy)
        return model(lr,noisy)
    if kind == 'ffdnet':
        s = noisy.new_full((noisy.shape[0], 1, 1, 1), float(sigma) / 255.0)
        return model(noisy, s)
    return model(noisy)

def sobel(x):
    kx=x.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]]).view(1,1,3,3); ky=x.new_tensor([[-1,-2,-1],[0,0,0],[1,2,1]]).view(1,1,3,3)
    gx=F.conv2d(x,kx,padding=1); gy=F.conv2d(x,ky,padding=1); mag=(gx.square()+gy.square()+1e-6).sqrt()
    return mag/(mag.amax((-2,-1),keepdim=True)+1e-6),gx,gy

class Stage1(nn.Module):
    def __init__(self, base=32):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(3,16,3,padding=1,bias=False),nn.GroupNorm(4,16),nn.ReLU(True),nn.Conv2d(16,base,3,padding=1,bias=False),nn.GroupNorm(8,base),nn.ReLU(True),nn.Conv2d(base,base,3,padding=1,bias=False),nn.GroupNorm(8,base),nn.ReLU(True),nn.Conv2d(base,2,1))
    def forward(self, raw, coarse):
        z=self.net(torch.cat([raw,coarse,sobel(coarse)[0]],1)); return z[:,:1]*torch.sigmoid(z[:,1:2])

class AB(nn.Module):
    def __init__(self, base=32, smooth_gate=False):
        super().__init__(); self.smooth_gate=smooth_gate; out=4 if smooth_gate else 3
        self.net=nn.Sequential(nn.Conv2d(4,16,3,padding=1,bias=False),nn.GroupNorm(4,16),nn.ReLU(True),nn.Conv2d(16,base,3,padding=1,bias=False),nn.GroupNorm(8,base),nn.ReLU(True),nn.Conv2d(base,base,3,padding=1,bias=False),nn.GroupNorm(8,base),nn.ReLU(True),nn.Conv2d(base,out,1))
    def forward(self, raw, coarse, prev):
        z=self.net(torch.cat([raw,coarse,sobel(coarse)[0],prev],1)); a=1+.2*torch.tanh(z[:,:1]); b=.2*torch.tanh(z[:,1:2])
        if self.smooth_gate:
            eg=torch.sigmoid(z[:,2:3]); sg=torch.sigmoid(z[:,3:4]); gate=eg*(1-sg)+.25*sg
        else: gate=torch.sigmoid(z[:,2:3])
        return a,b,gate

class Exp6Refine(nn.Module):
    def __init__(self, variant='exp6e'):
        super().__init__(); self.variant=variant; self.s1=Stage1(); self.s2=AB(smooth_gate=variant=='exp6d'); self.s3=AB(smooth_gate=variant=='exp6d')
    def forward(self, raw, coarse):
        d1=self.s1(raw,coarse); c1=coarse+d1; a2,b2,g2=self.s2(raw,c1,d1); d2=d1+g2*(a2*d1+b2-d1); c2=coarse+d2; a3,b3,g3=self.s3(raw,c2,d2); d3=d2+g3*(a3*d2+b3-d2)
        return {'delta1':d1,'delta2':d2,'delta3':d3,'final':coarse+d3}


class Exp7CRefine(Exp6Refine):
    """Gray-scale port of Exp7C's three-stage cascade refinement network."""
    def __init__(self):
        super().__init__('exp6e')
        self.variant = 'exp7c'

def otsu_threshold(g, bins=256):
    flat=g.flatten(1); gmin=flat.min(1).values; gmax=flat.max(1).values; span=(gmax-gmin).clamp_min(1e-6)
    idx=(((flat-gmin[:,None])/span[:,None]).clamp(0,1)*(bins-1)).long(); B=g.shape[0]
    offsets=(torch.arange(B,device=g.device)*bins)[:,None]; hist=torch.bincount((idx+offsets).reshape(-1),minlength=B*bins).reshape(B,bins).to(g.dtype)
    prob=hist/hist.sum(1,keepdim=True).clamp_min(1e-6); centers=torch.linspace(0,1,bins,device=g.device,dtype=g.dtype)[None]
    omega=prob.cumsum(1); mu=(prob*centers).cumsum(1); total=mu[:,-1:]; score=(total*omega-mu).square()/(omega*(1-omega)).clamp_min(1e-6); score[:,:1]=0; score[:,-1:]=0
    return gmin+centers[0,score.argmax(1)]*span

def edge_loss(pred,target,mask):
    edge, smooth, gx, gy = edge_statistics(target, mask)
    return edge_loss_from_statistics(pred, target, edge, smooth, gx, gy)


def edge_statistics(target, mask):
    mag,gx,gy=sobel(target); T=otsu_threshold(mag); edge=(mag>T[:,None,None,None]).float(); count=edge.flatten(1).sum(1); flat=edge.flatten(1); flat[count<=0]=0; flat[count<=0,mag.flatten(1)[count<=0].argmax(1)]=1; edge=flat.view_as(edge); edge=F.max_pool2d(edge,3,1,1)*mask
    return edge, mask-edge, gx, gy


def edge_loss_from_statistics(pred, target, edge, smooth, gx, gy):
    err=(pred-target).abs(); grad=(sobel(pred)[1:][0]-gx).abs()+(sobel(pred)[2:][0]-gy).abs()
    def red(x,m): return (x*m).sum()/(m.sum()+1e-6)
    return red(err,smooth)+2*red(err,edge)+.2*red(grad,edge)

def exp6_loss(out, coarse, clean, mask):
    target=clean-coarse
    edge, smooth, gx, gy = edge_statistics(target, mask)
    l1=edge_loss_from_statistics(out['delta1'],target,edge,smooth,gx,gy); l2=edge_loss_from_statistics(out['delta2'],target,edge,smooth,gx,gy); l3=edge_loss_from_statistics(out['delta3'],target,edge,smooth,gx,gy)
    return .1*l1+.3*l2+.6*l3, (l1,l2,l3)


def exp7c_loss(out, coarse, clean, mask):
    """Exp7C uses the same explicit edge/smooth weights as ``edge_loss``."""
    return exp6_loss(out, coarse, clean, mask)


def masked_smooth_l1(pred, target, mask):
    loss = F.smooth_l1_loss(pred, target, reduction='none') * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def masked_mse(pred, target, mask):
    loss = (pred - target).square() * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def coarse_loss(pred, target, mask, mode='smooth'):
    if mode == 'smooth':
        return .5 * masked_smooth_l1(pred, target, mask)
    if mode == 'mse':
        return masked_mse(pred, target, mask)
    if mode == 'mse_smooth':
        return masked_mse(pred, target, mask) + .5 * masked_smooth_l1(pred, target, mask)
    raise ValueError(f'Unknown coarse loss mode: {mode}')


def masked_psnr_per_image(prediction, target, mask):
    mse = ((prediction - target).square() * mask).sum((1, 2, 3)) / mask.sum((1, 2, 3)).clamp_min(1.0)
    return 10 * torch.log10(1 / mse)


def joint_loss(rf_loss, coarse, clean, mask):
    fidelity = masked_smooth_l1(coarse, clean, mask)
    return rf_loss + .5 * fidelity, fidelity

def load_denoiser(kind, checkpoint, device):
    root=Path(__file__).parent
    third_party = root / 'third_party'
    if str(third_party) not in sys.path:
        sys.path.insert(0, str(third_party))
    if kind=='convguided':
        from exp8.convguided_adapter import ConvGuidedGray, load_convguided
        model=ConvGuidedGray() if checkpoint is None else load_convguided(checkpoint,'cpu')
    elif kind=='dncnn':
        from DnCNN.refine_bridge.denoiser import DnCNN
        model=DnCNN()
        old=getattr(sys.modules['__main__'],'DnCNN',None); sys.modules['__main__'].DnCNN=DnCNN
        try: obj=torch.load(checkpoint,map_location='cpu',weights_only=False)
        finally:
            if old is None: delattr(sys.modules['__main__'],'DnCNN')
            else: sys.modules['__main__'].DnCNN=old
        if isinstance(obj,nn.Module): model=obj
        else: model.load_state_dict(obj.get('model_state_dict', obj.get('state_dict', obj)),strict=True)
    elif kind=='ffdnet':
        from FFDNet.refine_bridge.denoiser import load_ffdnet
        model=load_ffdnet(checkpoint,'cpu')
    else: raise ValueError(kind)
    return model.to(device)

def unwrap_state(m): return m.module.state_dict() if isinstance(m,nn.DataParallel) else m.state_dict()

def save_bundle(path, denoiser, refine, metrics, epoch):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'denoiser':unwrap_state(denoiser),'refine':unwrap_state(refine),'metrics':metrics,'epoch':epoch},path)
    (path.parent/'metrics.json').write_text(json.dumps(metrics,indent=2))
