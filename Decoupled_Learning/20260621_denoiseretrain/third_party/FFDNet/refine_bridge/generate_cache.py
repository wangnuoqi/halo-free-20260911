from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
import time
import argparse, torch
import numpy as np
from refine_bridge_common import image_files, load_gray, save_cache, SIGMA
from denoiser import load_ffdnet

def make_split(root, model, sigma, seed0, device):
    out=[]
    with torch.no_grad():
        files=image_files(root); total=len(files); start=time.time()
        for i,p in enumerate(files):
            clean=load_gray(p).to(device)
            # KAIR's official BSD68 test code resets NumPy seed=0 for each image.
            rng=np.random.RandomState(0 if seed0 else i)
            noise=torch.from_numpy(rng.normal(0, sigma/255., clean.shape).astype(np.float32)).to(device)
            noisy=clean+noise; sigma_tensor=torch.full((1,1,1,1),sigma/255.,device=device,dtype=clean.dtype); coarse=model(noisy,sigma_tensor).clamp(0,1)
            out.append({'name':p.name,'clean':clean.cpu(),'noisy':noisy.cpu(),'coarse':coarse.cpu()})
            if (i+1)%10==0 or i+1==total: print(f'cache {i+1}/{total} ({time.time()-start:.1f}s)',flush=True)
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train-root',required=True); ap.add_argument('--test-root',required=True); ap.add_argument('--checkpoint',default=str(Path(__file__).parent/'pytorch_official/models/ffdnet_gray.pth')); ap.add_argument('--out',required=True); ap.add_argument('--sigma',type=float,default=SIGMA); ap.add_argument('--device',default='cpu'); args=ap.parse_args()
    dev=torch.device(args.device); model=load_ffdnet(args.checkpoint,dev); cache={'train':make_split(args.train_root,model,args.sigma,0,dev),'test':make_split(args.test_root,model,args.sigma,100000,dev),'sigma':args.sigma,'denoiser':'FFDNet-KAIR-official-PyTorch'}; save_cache(Path(args.out),cache)
if __name__=='__main__': main()
