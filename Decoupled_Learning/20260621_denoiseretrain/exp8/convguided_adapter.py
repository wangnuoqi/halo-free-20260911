"""Self-contained loader for the existing grayscale ConvGuided denoiser."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveNorm(nn.Module):
    def __init__(self,n):
        super().__init__(); self.w_0=nn.Parameter(torch.tensor([1.])); self.w_1=nn.Parameter(torch.tensor([0.])); self.bn=nn.BatchNorm2d(n,momentum=.999,eps=.001)
    def forward(self,x): return self.w_0*x+self.w_1*self.bn(x)


def weights_init_identity(module):
    if isinstance(module, nn.Conv2d):
        out_channels, in_channels, height, width = module.weight.shape
        if out_channels < in_channels:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: nn.init.zeros_(module.bias)
            return
        module.weight.data.zero_()
        center_height, center_width = height // 2, width // 2
        for channel in range(in_channels):
            module.weight.data[channel, channel, center_height, center_width] = 1.0
        if module.bias is not None: nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight); nn.init.zeros_(module.bias)

class ConvGuidedFilter(nn.Module):
    def __init__(self,radius=1):
        super().__init__(); self.box_filter=nn.Conv2d(1,1,3,padding=radius,dilation=radius,bias=False)
        self.conv_a=nn.Sequential(nn.Conv2d(2,32,1,bias=False),AdaptiveNorm(32),nn.ReLU(True),nn.Conv2d(32,32,1,bias=False),AdaptiveNorm(32),nn.ReLU(True),nn.Conv2d(32,1,1,bias=False)); self.box_filter.weight.data.fill_(1.)
    def forward(self,x_lr,y_lr,x_hr):
        h,w=x_lr.shape[-2:]; count=self.box_filter(x_lr.new_ones((1,1,h,w))); mx=self.box_filter(x_lr)/count; my=self.box_filter(y_lr)/count; cov=self.box_filter(x_lr*y_lr)/count-mx*my; var=self.box_filter(x_lr*x_lr)/count-mx*mx; a=self.conv_a(torch.cat([cov,var],1)); b=my-a*mx; ma=F.interpolate(a,x_hr.shape[-2:],mode='bilinear',align_corners=True); mb=F.interpolate(b,x_hr.shape[-2:],mode='bilinear',align_corners=True); return ma*x_hr+mb

class ConvGuidedGray(nn.Module):
    def __init__(self):
        super().__init__(); layers=[nn.Conv2d(1,24,3,padding=1,bias=False),AdaptiveNorm(24),nn.LeakyReLU(.2,True)]
        for level in range(1,5): layers += [nn.Conv2d(24,24,3,padding=2**level,dilation=2**level,bias=False),AdaptiveNorm(24),nn.LeakyReLU(.2,True)]
        layers += [nn.Conv2d(24,24,3,padding=1,bias=False),AdaptiveNorm(24),nn.LeakyReLU(.2,True),nn.Conv2d(24,1,1)]
        self.lr=nn.Sequential(*layers); self.lr.apply(weights_init_identity); self.gf=ConvGuidedFilter()
    def forward(self,x_lr,x_hr): return self.gf(x_lr,self.lr(x_lr),x_hr).clamp(0,1)

def load_convguided(checkpoint,device='cpu'):
    model=ConvGuidedGray(); state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if isinstance(state,dict) and 'model_state_dict' in state: state=state['model_state_dict']
    if isinstance(state,dict): state={k[7:] if k.startswith('module.') else k:v for k,v in state.items()}
    model.load_state_dict(state,strict=True); return model.to(device)
