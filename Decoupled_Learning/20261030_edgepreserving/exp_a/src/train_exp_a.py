"""Experiment A: from-scratch joint ConvGuided + 3-stage training.

Variants:
  c0: original LR predictor + legacy cascade loss
  c1: VIV-inspired LR predictor + legacy cascade loss
  a0: original LR predictor + unified image-space edge loss
  a1: VIV-inspired LR predictor + unified image-space edge loss

The guided-filter operator and the delta/a/b/gate recurrence are shared by all
variants. No old checkpoint is loaded by this trainer.
"""
import argparse, json, math, os, random, contextlib
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def seed_all(seed, rank=0):
    value=seed+rank
    random.seed(value)
    np.random.seed(value)
    # torch.manual_seed() also calls cuda.manual_seed_all() in PyTorch; after
    # set_device(local_rank), seed only the CPU generator explicitly instead.
    torch.random.default_generator.manual_seed(value)
    # Handoff protocol: seed only the rank's already-bound CUDA device.
    # Do not call manual_seed_all(), which can initialize every visible GPU.
    torch.cuda.manual_seed(value)


def init_parallel(mode):
    if mode == 'dp':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable for DataParallel')
        visible = torch.cuda.device_count()
        if visible != 4:
            raise RuntimeError(f'Expected four visible CUDA GPUs for DataParallel, found {visible}')
        return False, 0, visible, torch.device('cuda:0'), None
    local_rank=int(os.environ.get('LOCAL_RANK','-1'))
    if local_rank < 0:
        return False, 0, 1, torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), None
    visible=torch.cuda.device_count(); world=int(os.environ['WORLD_SIZE'])
    if visible < world: raise RuntimeError(f'requested {world} ranks, but only {visible} CUDA devices are visible')
    torch.cuda.set_device(local_rank)
    backend = os.environ.get('TORCH_DDP_BACKEND', 'nccl')
    if os.name == 'nt' and backend == 'nccl':
        backend = 'gloo'
    dist.init_process_group(backend=backend, init_method='env://')
    return True, int(os.environ['RANK']), world, torch.device('cuda',local_rank), backend


class Pairs(Dataset):
    def __init__(self, root, split):
        tail = "-200" if split == "test" else ""
        root = Path(root)
        self.lr = sorted((root / f"{split}input-l0p500-a1p400{tail}_downsample").glob("*.tif*"))
        self.hr = sorted((root / f"{split}input-l0p500-a1p400{tail}").glob("*.tif*"))
        self.y = sorted((root / f"{split}label-l0p500-a1p400{tail}").glob("*.tif*"))
        if not self.lr or not (len(self.lr) == len(self.hr) == len(self.y)): raise ValueError("unmatched WLS triples")
    def __len__(self): return len(self.hr)
    def __getitem__(self, i):
        lr = cv2.imread(str(self.lr[i]), -1).astype("float32")
        x = cv2.imread(str(self.hr[i]), -1).astype("float32")
        y = cv2.imread(str(self.y[i]), -1).astype("float32")
        if lr.ndim != 2 or x.ndim != 2 or x.shape != y.shape: raise ValueError("expected paired grayscale TIFF triple")
        return torch.from_numpy(lr[None]), torch.from_numpy(x[None]), torch.from_numpy(y[None])


def collate(batch, crop):
    h = min(x.shape[-2] for _, x, _ in batch); w = min(x.shape[-1] for _, x, _ in batch)
    if crop:
        outlr=[]; outx=[]; outy=[]
        for lr,x,y in batch:
            hh,ww=x.shape[-2:]
            scale=hh//lr.shape[-2]
            if hh % lr.shape[-2] != 0 or scale < 1 or ww % lr.shape[-1] != 0 or ww // lr.shape[-1] != scale:
                raise ValueError("HR/LR scale must be equal and integral in both dimensions")
            if crop % scale != 0 or crop > hh or crop > ww:
                raise ValueError(f"crop size {crop} must fit HR image and be divisible by scale {scale}")
            top=random.randint(0,(hh-crop)//scale)*scale
            left=random.randint(0,(ww-crop)//scale)*scale
            lc=crop//scale; outlr.append(lr[...,top//scale:top//scale+lc,left//scale:left//scale+lc])
            outx.append(x[...,top:top+crop,left:left+crop]); outy.append(y[...,top:top+crop,left:left+crop])
        return torch.stack(outlr),torch.stack(outx),torch.stack(outy)
    h -= h % 2; w -= w % 2
    return (torch.stack([lr for lr,_,_ in batch]), torch.stack([x[...,:h,:w] for _,x,_ in batch]), torch.stack([y[...,:h,:w] for _,_,y in batch]))


class PairedPatchCollator:
    """Pickleable DataLoader collator for Windows multiprocessing workers."""
    def __init__(self, crop):
        self.crop = int(crop)

    def __call__(self, batch):
        return collate(batch, self.crop)


class AdaptiveNorm(nn.Module):
    def __init__(self,n):
        super().__init__(); self.w0=nn.Parameter(torch.ones(1)); self.w1=nn.Parameter(torch.zeros(1)); self.bn=nn.BatchNorm2d(n,momentum=.999,eps=.001)
    def forward(self,x): return self.w0*x+self.w1*self.bn(x)


class ConvGuidedFilter(nn.Module):
    def __init__(self):
        super().__init__(); self.box=nn.Conv2d(1,1,3,padding=1,bias=False); self.box.weight.data.fill_(1.)
        self.a=nn.Sequential(nn.Conv2d(2,32,1,bias=False),AdaptiveNorm(32),nn.ReLU(),nn.Conv2d(32,32,1,bias=False),AdaptiveNorm(32),nn.ReLU(),nn.Conv2d(32,1,1,bias=False))
    def forward(self,x,y,guide):
        n=self.box(torch.ones_like(x)); mx=self.box(x)/n; my=self.box(y)/n
        cov=self.box(x*y)/n-mx*my; var=self.box(x*x)/n-mx*mx
        a=self.a(torch.cat([cov,var],1)); b=my-a*mx
        ah=F.interpolate(a,guide.shape[-2:],mode='bilinear',align_corners=True); bh=F.interpolate(b,guide.shape[-2:],mode='bilinear',align_corners=True)
        return (ah*guide+bh).clamp(0,1)


class OriginalLR(nn.Module):
    def __init__(self):
        super().__init__(); layers=[nn.Conv2d(1,24,3,padding=1,bias=False),AdaptiveNorm(24),nn.LeakyReLU(.2)]
        for level in range(1,5): layers += [nn.Conv2d(24,24,3,padding=2**level,dilation=2**level,bias=False),AdaptiveNorm(24),nn.LeakyReLU(.2)]
        layers += [nn.Conv2d(24,24,3,padding=1,bias=False),AdaptiveNorm(24),nn.LeakyReLU(.2),nn.Conv2d(24,1,1)]
        self.net=nn.Sequential(*layers); self.net.apply(self._identity_init)
    @staticmethod
    def _identity_init(m):
        if isinstance(m,nn.Conv2d):
            if m.out_channels < m.in_channels:
                nn.init.xavier_uniform_(m.weight); return
            m.weight.data.zero_(); h,w=m.kernel_size[0]//2,m.kernel_size[1]//2
            for i in range(min(m.out_channels,m.in_channels)): m.weight.data[i,i,h,w]=1.
        elif isinstance(m,nn.BatchNorm2d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    def forward(self,x): return self.net(x)


class VIVLR(nn.Module):
    """Small VIV-inspired predictor: 3x3/9x9, correlation weighting, order-4 products."""
    def __init__(self, dim=24, bounded_output=False):
        super().__init__(); self.inp=nn.Conv2d(1,dim*2,3,padding=1); self.dw3=nn.Conv2d(dim,dim,3,padding=1,groups=dim); self.dw9=nn.Conv2d(dim,dim,9,padding=4,groups=dim)
        self.bounded_output=bounded_output
        mid=max(1,round(.6*dim*2)); self.weight=nn.Sequential(nn.Linear(dim*2,mid),nn.LeakyReLU(),nn.Linear(mid,1)); self.out=nn.Conv2d(dim,1,1)
        if bounded_output:
            nn.init.normal_(self.out.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.out.bias)
    def forward(self,x):
        f3,f9=self.inp(x).chunk(2,1); f3=self.dw3(f3); f9=self.dw9(f9); f=torch.cat([f3,f9],1)
        z=f.permute(0,2,3,1).reshape(f.shape[0],-1,f.shape[1]); z=F.normalize(z,dim=1); cov=z.transpose(1,2)@z
        w=self.weight(cov).squeeze(-1).unsqueeze(-1).unsqueeze(-1); w3,w9=w.chunk(2,1); f3=f3*(1+w3); f9=f9*(1+w9)
        z=f3*f9; z=F.conv2d(z, self._kernel(self.dw3), padding=1, groups=z.shape[1]) * f9
        z=F.conv2d(z, self._kernel(self.dw3), padding=1, groups=z.shape[1]) * f3
        output=self.out(z)
        # Transfer runs preserve the old LR predictor's [0, 1] interface.
        return torch.sigmoid(output) if self.bounded_output else output
    @staticmethod
    def _kernel(layer): return layer.weight


def sobel(x):
    kx=x.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]]).view(1,1,3,3); ky=x.new_tensor([[-1,-2,-1],[0,0,0],[1,2,1]]).view(1,1,3,3)
    gx=F.conv2d(x,kx,padding=1); gy=F.conv2d(x,ky,padding=1); return torch.sqrt(gx*gx+gy*gy+1e-6),gx,gy


def normalized_sobel_magnitude(x):
    magnitude,_,_=sobel(x)
    return magnitude/(magnitude.amax(dim=(2,3),keepdim=True)+1e-6)


def otsu_threshold(g, bins=256, eps=1e-6):
    flat=g.flatten(1); lo=flat.min(1).values; hi=flat.max(1).values; span=(hi-lo).clamp_min(eps)
    norm=((flat-lo[:,None])/span[:,None]).clamp(0,1); idx=(norm*(bins-1)).long(); offsets=torch.arange(g.shape[0],device=g.device)[:,None]*bins
    hist=torch.bincount((idx+offsets).reshape(-1),minlength=g.shape[0]*bins).reshape(g.shape[0],bins).to(g.dtype); prob=hist/hist.sum(1,keepdim=True).clamp_min(eps)
    centers=torch.linspace(0,1,bins,device=g.device,dtype=g.dtype)[None,:]; omega=prob.cumsum(1); mu=(prob*centers).cumsum(1); total=mu[:,-1:]
    score=((total*omega-mu)**2)/(omega*(1-omega)).clamp_min(eps); score[:,0]=0; score[:,-1]=0
    t=centers[0,score.argmax(1)]; t=lo+t*span; return torch.where(hi-lo>eps,torch.minimum(t,hi-eps),hi)


class NAFBlock(nn.Module):
    def __init__(self,c):
        super().__init__(); self.norm=nn.GroupNorm(max(1,min(8,c)),c); self.p=nn.Conv2d(c,c*2,1); self.dw=nn.Conv2d(c*2,c*2,3,padding=1,groups=c*2); self.sca=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(c,c,1)); self.o=nn.Conv2d(c,c,1); self.beta=nn.Parameter(torch.zeros(1))
    def forward(self,x):
        z=self.dw(self.p(self.norm(x))); a,b=z.chunk(2,1); z=a*b; z=z*self.sca(z); return x+self.beta*self.o(z)


class PlainBlock(nn.Module):
    def __init__(self,c):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(c,c,3,padding=1,bias=False),nn.GroupNorm(max(1,min(8,c)),c),nn.ReLU(),nn.Conv2d(c,c,3,padding=1,bias=False),nn.GroupNorm(max(1,min(8,c)),c),nn.ReLU())
    def forward(self,x): return x+self.net(x)


class OriginalStage(nn.Module):
    def __init__(self,outc,base=32):
        super().__init__(); self.inp=nn.Sequential(nn.Conv2d(3 if outc==2 else 4,16,3,padding=1,bias=False),nn.GroupNorm(4,16),nn.ReLU(),nn.Conv2d(16,base,3,padding=1,bias=False),nn.GroupNorm(8,base),nn.ReLU(),nn.Conv2d(base,base,3,padding=1,bias=False),nn.GroupNorm(8,base),nn.ReLU(),nn.Conv2d(base,outc,1))
    def forward(self,x,coarse,delta=None):
        g=normalized_sobel_magnitude(coarse); return self.inp(torch.cat([x,coarse,g] + ([] if delta is None else [delta]),1))


class Stage(nn.Module):
    def __init__(self,outc,base=32,use_naf=False):
        super().__init__(); self.inp=nn.Conv2d(3 if outc==2 else 4,16,3,padding=1); block=NAFBlock if use_naf else PlainBlock; self.blocks=nn.Sequential(block(16),nn.Conv2d(16,base,3,padding=1),block(base),nn.Conv2d(base,base,3,padding=1),block(base),nn.Conv2d(base,outc,1))
    def forward(self,x,coarse,delta=None):
        g=normalized_sobel_magnitude(coarse); z=torch.cat([x,coarse,g] + ([] if delta is None else [delta]),1); return self.blocks(self.inp(z))


class Cascade(nn.Module):
    def __init__(self,use_naf=False):
        super().__init__(); cls=Stage if use_naf else OriginalStage; self.s1=cls(2); self.s2=cls(3); self.s3=cls(3)
    def forward(self,x,c):
        q=self.s1(x,c); d1=q[:,:1]*torch.sigmoid(q[:,1:2]); c1=c+d1
        q=self.s2(x,c1,d1); a=1+.2*torch.tanh(q[:,:1]); b=.2*torch.tanh(q[:,1:2]); g=torch.sigmoid(q[:,2:3]); d2=d1+g*(a*d1+b-d1); c2=c+d2
        q=self.s3(x,c2,d2); a=1+.2*torch.tanh(q[:,:1]); b=.2*torch.tanh(q[:,1:2]); g=torch.sigmoid(q[:,2:3]); d3=d2+g*(a*d2+b-d2)
        return {'coarse1':c1,'coarse2':c2,'final':c+d3,'delta1':d1,'delta2':d2,'delta3':d3}


class Full(nn.Module):
    def __init__(self, use_viv, use_naf=False, bounded_viv_output=False):
        super().__init__(); self.lr=VIVLR(bounded_output=bounded_viv_output) if use_viv else OriginalLR(); self.gf=ConvGuidedFilter(); self.ref=Cascade(use_naf=use_naf)
    def forward(self, lr, x):
        # Inputs lr and x are already centered by the caller, matching exp7C.
        c=self.gf(lr,self.lr(lr),x); r=self.ref(x,c-.5)
        o={k:(v+.5).clamp(0,1) for k,v in r.items() if k in ('coarse1','coarse2','final')}
        o.update({k:v for k,v in r.items() if k in ('delta1','delta2','delta3')}); o['coarse0']=c; return o


def edge_loss(pred,target):
    g,gx,gy=sobel(target); edge=(g>otsu_threshold(g).view(-1,1,1,1)).float(); counts=edge.flatten(1).sum(1); no_edge=counts<=0
    if no_edge.any():
        flat=edge.new_zeros(edge.flatten(1).shape); flat.scatter_(1,g.flatten(1).argmax(1,keepdim=True),1); m=edge.flatten(1); m[no_edge]=flat[no_edge]; edge=m.view_as(edge)
    edge=edge.detach(); smooth=(1-edge).clamp(0,1); pix=(pred-target).abs(); _,px,py=sobel(pred); grad=(px-gx).abs()+(py-gy).abs()
    return (smooth*pix).sum()/smooth.sum().clamp_min(1)+(2*edge*pix).sum()/edge.sum().clamp_min(1)+.2*(edge*grad).sum()/edge.sum().clamp_min(1)


def legacy_loss(outputs, coarse0, target):
    """Original exp1/exp7 cascade form: edge loss is on delta and target delta."""
    target_delta = target - coarse0
    l1=edge_loss(outputs['delta1'], target_delta)
    l2=edge_loss(outputs['delta2'], target_delta)
    l3=edge_loss(outputs['delta3'], target_delta)
    return .10*l1+.30*l2+.60*l3, (l1,l2,l3)


def load_exp7c_components(model, guided_path, refine_path):
    """Load the trained guided-filter and three stages, leaving the new LR net intact."""
    guided = torch.load(guided_path, map_location='cpu', weights_only=False)
    refine = torch.load(refine_path, map_location='cpu', weights_only=False)
    target = model.state_dict()
    loaded = {}

    for key, value in guided.items():
        if key.startswith('gf.box_filter.'):
            new_key = key.replace('gf.box_filter.', 'gf.box.', 1)
        elif key.startswith('gf.conv_a.'):
            new_key = key.replace('gf.conv_a.', 'gf.a.', 1)
        else:
            continue
        new_key = new_key.replace('.w_0', '.w0').replace('.w_1', '.w1')
        if new_key not in target or target[new_key].shape != value.shape:
            raise ValueError(f'Cannot map exp7C guided parameter {key} -> {new_key}')
        loaded[new_key] = value

    refine_state = refine.get('state_dict', refine)
    mapped_refine = {}
    for key, value in refine_state.items():
        if key.startswith('module.'):
            key = key[len('module.'):]
        if key.startswith('stage1.net.'):
            new_key = key.replace('stage1.net.', 'ref.s1.inp.', 1)
        elif key.startswith('stage2.net.'):
            new_key = key.replace('stage2.net.', 'ref.s2.inp.', 1)
        elif key.startswith('stage3.net.'):
            new_key = key.replace('stage3.net.', 'ref.s3.inp.', 1)
        else:
            raise ValueError(f'Unexpected exp7C refine key: {key}')
        if new_key not in target or target[new_key].shape != value.shape:
            raise ValueError(f'Cannot map exp7C refine parameter {key} -> {new_key}')
        mapped_refine[new_key] = value
    loaded.update(mapped_refine)

    expected = {key for key in target if key.startswith('gf.') or key.startswith('ref.')}
    missing = sorted(expected - loaded.keys())
    extra = sorted(loaded.keys() - expected)
    if missing or extra:
        raise ValueError(f'Incomplete exp7C initialization: missing={missing}, unexpected={extra}')
    target.update(loaded)
    model.load_state_dict(target, strict=True)
    return len(loaded), len(expected)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--variant',choices=['c0','c1','a0','a1'],required=True); ap.add_argument('--data-root',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--crop-size',type=int,default=128); ap.add_argument('--batch-size-per-gpu','--batch-size',type=int,default=4); ap.add_argument('--epochs',type=int,default=200); ap.add_argument('--lr',type=float,default=3e-4); ap.add_argument('--finetune-lr',type=float,default=3e-5); ap.add_argument('--min-lr',type=float,default=3e-6); ap.add_argument('--lr-schedule',choices=['cosine','constant'],default='cosine'); ap.add_argument('--precision',choices=['fp32','bf16'],default='bf16'); ap.add_argument('--num-workers',type=int,default=4); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--device',default=None,help='单卡兼容参数；DDP 时由 LOCAL_RANK 决定'); ap.add_argument('--parallelism',choices=['auto','ddp','dp'],default='auto'); ap.add_argument('--init-exp7c',action='store_true',help='Load exp7C ConvGuided and 3-stage weights; keep the selected LR predictor newly initialized'); ap.add_argument('--guided-checkpoint',default=None); ap.add_argument('--refine-checkpoint',default=None); ap.add_argument('--max-steps',type=int,default=0,help='仅用于 smoke test；0 表示完整训练'); args=ap.parse_args()
    if args.init_exp7c and args.variant != 'c1': ap.error('--init-exp7c is only valid for variant c1')
    mode=args.parallelism
    if mode=='auto': mode='ddp' if 'LOCAL_RANK' in os.environ else 'dp'
    is_ddp,rank,world,d,backend=init_parallel(mode); seed_all(args.seed,rank); is_main=(rank==0)
    net=Full(args.variant in ('c1','a1'),use_naf=args.variant=='a1',bounded_viv_output=args.init_exp7c).to(d)
    if args.init_exp7c:
        root=Path(__file__).resolve().parents[1] / 'checkpoints' / 'c0'
        guided_path=args.guided_checkpoint or str(root / 'convguided_joint_best_20260818_l0p500_a1p400.pth')
        refine_path=args.refine_checkpoint or str(root / 'rf_3stage_best_20260818_l0p500_a1p400.pth')
        if not Path(guided_path).is_file() or not Path(refine_path).is_file():
            ap.error(f'exp7C initialization checkpoints not found: {guided_path}; {refine_path}')
        args.guided_checkpoint=str(Path(guided_path).resolve()); args.refine_checkpoint=str(Path(refine_path).resolve())
        loaded_count,expected_count=load_exp7c_components(net,guided_path,refine_path)
        if is_main: print(f'Loaded exp7C ConvGuided + 3-stage: {loaded_count}/{expected_count} tensors; VIV LR predictor initialized for fine-tuning',flush=True)
    if mode=='ddp': net=DDP(net,device_ids=[d.index],find_unused_parameters=False)
    elif mode=='dp': net=torch.nn.DataParallel(net,device_ids=list(range(world)))
    base_model=net.module if is_ddp or mode=='dp' else net
    if args.init_exp7c:
        opt=torch.optim.AdamW([
            {'params':base_model.lr.parameters(),'lr':args.lr},
            {'params':list(base_model.gf.parameters())+list(base_model.ref.parameters()),'lr':args.finetune_lr},
        ],weight_decay=1e-4)
    else:
        opt=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs), eta_min=args.min_lr) if args.lr_schedule == 'cosine' else None
    ds=Pairs(args.data_root,'train'); sampler=DistributedSampler(ds,num_replicas=world,rank=rank,shuffle=True,seed=args.seed) if is_ddp else None; loader_batch=args.batch_size_per_gpu if is_ddp else args.batch_size_per_gpu*world; dl=DataLoader(ds,batch_size=loader_batch,shuffle=sampler is None,sampler=sampler,num_workers=args.num_workers,pin_memory=False,persistent_workers=args.num_workers>0,collate_fn=PairedPatchCollator(args.crop_size)); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); protocol=vars(args)|{'variant':args.variant,'from_scratch':not args.init_exp7c,'initialization':'exp7C ConvGuided+3-stage; new VIV LR' if args.init_exp7c else 'random','lr_output_mode':'sigmoid_0_1' if args.init_exp7c else 'raw','parallelism':'DDP' if is_ddp else 'DataParallel','ddp':is_ddp,'backend':backend,'world_size':world,'global_batch_size':args.batch_size_per_gpu*world,'lr_predictor':'viv' if args.variant in ('c1','a1') else 'original','stage_blocks':'naf' if args.variant=='a1' else 'original','loss_mode':'unified' if args.variant in ('a0','a1') else 'legacy','train_geometry':'random_paired_patch','inference_geometry':'full_image'}; (out/'config.json').write_text(json.dumps(protocol,indent=2)) if is_main else None
    if is_main: print(f'Experiment A {args.variant}: mode={"DDP" if is_ddp else "DataParallel"}, backend={backend or "none"}, world={world}, batch/GPU={args.batch_size_per_gpu}, global batch={args.batch_size_per_gpu*world}, crop={args.crop_size}, epochs={args.epochs}, lr={args.lr}->{args.min_lr} ({args.lr_schedule}), precision={args.precision}',flush=True)
    step=0; completed_epochs=0
    for ep in range(args.epochs):
        net.train()
        if sampler is not None: sampler.set_epoch(ep)
        for lr,x,y in dl:
            lr=lr.to(d); x=x.to(d); y=y.to(d)
            amp=torch.autocast('cuda',dtype=torch.bfloat16) if args.precision=='bf16' and d.type=='cuda' else contextlib.nullcontext()
            with amp:
                outp=net(lr-.5, x-.5)
                if args.variant in ('a0','a1'):
                    losses=[edge_loss(outp[k],y) for k in ('coarse0','coarse1','coarse2','final')]
                    loss=.05*losses[0]+.10*losses[1]+.30*losses[2]+.60*losses[3]
                else:
                    cascade,_=legacy_loss(outp,outp['coarse0'],y)
                    # Preserve the historical exp7C joint objective: legacy
                    # delta cascade plus coarse fidelity.
                    loss=cascade+0.5*F.smooth_l1_loss(outp['coarse0'],y)
            opt.zero_grad(); loss.backward(); opt.step(); step+=1
            if is_main and (step==1 or step%50==0): print(f'variant={args.variant} epoch={ep+1} step={step} loss={loss.item():.6f} lr={[group["lr"] for group in opt.param_groups]}',flush=True)
            if args.max_steps and step >= args.max_steps: break
        if scheduler is not None: scheduler.step()
        completed_epochs=ep+1
        if is_main: torch.save({'model_state':net.module.state_dict() if is_ddp or mode=='dp' else net.state_dict(),'step':step,'epoch':ep+1,'lr':opt.param_groups[0]['lr'],'run_protocol':protocol},out/'latest.pth')
        if args.max_steps and step >= args.max_steps: break
    if is_main: torch.save({'model_state':net.module.state_dict() if is_ddp or mode=='dp' else net.state_dict(),'step':step,'epoch':completed_epochs,'lr':opt.param_groups[0]['lr'],'run_protocol':protocol},out/'final.pth')
    if is_ddp: dist.destroy_process_group()

if __name__=='__main__': main()
