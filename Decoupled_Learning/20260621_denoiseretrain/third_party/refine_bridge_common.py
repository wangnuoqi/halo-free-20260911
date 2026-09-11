"""Shared full-image refine bridge utilities; independent of project code."""
from __future__ import annotations
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F

SIGMA = 25.0

def image_files(root):
    return sorted(p for p in Path(root).iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp'})

def load_gray(path):
    x = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if x is None: raise FileNotFoundError(path)
    h, w = x.shape
    return torch.from_numpy((x[:h-h%2, :w-w%2].astype(np.float32)/255.0)[None,None])

def add_awgn(clean, sigma=SIGMA, seed=None):
    g = torch.Generator(device=clean.device)
    if seed is not None: g.manual_seed(int(seed))
    return clean + torch.randn(clean.shape, generator=g, device=clean.device, dtype=clean.dtype) * (sigma/255.0)

def psnr(pred, target):
    mse = torch.mean((pred.clamp(0,1)-target)**2).item()
    return float('inf') if mse == 0 else 10.0*np.log10(1.0/mse)

def sobel(x, normalize=True):
    kx=x.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]]).view(1,1,3,3); ky=x.new_tensor([[-1,-2,-1],[0,0,0],[1,2,1]]).view(1,1,3,3)
    gx=F.conv2d(x,kx,padding=1); gy=F.conv2d(x,ky,padding=1); mag=(gx.square()+gy.square()+1e-6).sqrt()
    if normalize: mag=mag/(mag.amax(dim=(-2,-1),keepdim=True)+1e-6)
    return mag,gx,gy

class Stage1DeltaNet(nn.Module):
    def __init__(self, norm='group', base=32):
        super().__init__(); N=(lambda c: nn.GroupNorm(8 if c>=32 else 4,c)) if norm=='group' else (lambda c: nn.BatchNorm2d(c))
        self.net=nn.Sequential(nn.Conv2d(3,16,3,padding=1,bias=False),N(16),nn.ReLU(True),nn.Conv2d(16,base,3,padding=1,bias=False),N(base),nn.ReLU(True),nn.Conv2d(base,base,3,padding=1,bias=False),N(base),nn.ReLU(True),nn.Conv2d(base,1,1,bias=False))
    def forward(self, raw, coarse): return self.net(torch.cat([raw,coarse,sobel(coarse)[0]],1))

class ABAdjustNet(nn.Module):
    def __init__(self, norm='group', base=32, scale=.2):
        super().__init__(); N=(lambda c: nn.GroupNorm(8 if c>=32 else 4,c)) if norm=='group' else (lambda c: nn.BatchNorm2d(c))
        self.scale=scale; self.net=nn.Sequential(nn.Conv2d(4,16,3,padding=1,bias=False),N(16),nn.ReLU(True),nn.Conv2d(16,base,3,padding=1,bias=False),N(base),nn.ReLU(True),nn.Conv2d(base,base,3,padding=1,bias=False),N(base),nn.ReLU(True),nn.Conv2d(base,2,1))
    def forward(self, raw, coarse, prev):
        ab=self.net(torch.cat([raw,coarse,sobel(coarse)[0],prev],1)); return 1+self.scale*torch.tanh(ab[:,:1]), self.scale*torch.tanh(ab[:,1:])

class RefineCascade(nn.Module):
    def __init__(self, variant='groupnorm'):
        super().__init__(); norm='group' if variant=='groupnorm' else 'batch'; self.s1=Stage1DeltaNet(norm); self.s2=ABAdjustNet(norm); self.s3=ABAdjustNet(norm)
    def forward(self, raw, coarse):
        d1=self.s1(raw,coarse); c1=coarse+d1; a2,b2=self.s2(raw,c1,d1); d2=a2*d1+b2; c2=coarse+d2; a3,b3=self.s3(raw,c2,d2); d3=a3*d2+b3
        return {'delta1':d1,'delta2':d2,'delta3':d3,'final':coarse+d3}

def refine_loss(out, coarse, clean):
    target=clean-coarse
    return sum(F.l1_loss(out[k],target) for k in ('delta1','delta2','delta3'))/3 + .2*F.mse_loss(out['final'],clean)

def save_cache(path, records):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(records,tmp)
    tmp.replace(path)
