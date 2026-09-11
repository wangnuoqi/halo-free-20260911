from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
import time
import argparse, torch
from refine_bridge_common import image_files, load_gray, add_awgn, save_cache, SIGMA
from denoiser import load_dncnn

def make_split(root, model, sigma, seed0, device):
    out=[]
    with torch.no_grad():
        files=image_files(root); total=len(files); start=time.time()
        for i,p in enumerate(files):
            clean=load_gray(p).to(device); noisy=add_awgn(clean,sigma,seed0+i); coarse=model(noisy).clamp(0,1)
            out.append({'name':p.name,'clean':clean.cpu(),'noisy':noisy.cpu(),'coarse':coarse.cpu()})
            if (i+1)%10==0 or i+1==total: print(f'cache {i+1}/{total} ({time.time()-start:.1f}s)',flush=True)
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train-root',required=True); ap.add_argument('--test-root',required=True); ap.add_argument('--checkpoint',default=str(Path(__file__).parents[1]/'TrainingCodes/dncnn_pytorch/models/DnCNN_sigma25/model.pth')); ap.add_argument('--out',required=True); ap.add_argument('--sigma',type=float,default=SIGMA); ap.add_argument('--device',default='cpu'); args=ap.parse_args()
    dev=torch.device(args.device); model=load_dncnn(args.checkpoint,dev); cache={'train':make_split(args.train_root,model,args.sigma,0,dev),'test':make_split(args.test_root,model,args.sigma,100000,dev),'sigma':args.sigma,'denoiser':'DnCNN'}; save_cache(Path(args.out),cache)
if __name__=='__main__': main()
