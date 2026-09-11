import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parents[2]))
from refine_bridge_common import RefineCascade, refine_loss, psnr

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache',required=True); ap.add_argument('--variant',choices=['groupnorm','original'],required=True); ap.add_argument('--out',required=True); ap.add_argument('--epochs',type=int,default=80); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--batch-size',type=int,default=4); ap.add_argument('--device',default='cpu'); args=ap.parse_args()
    dev=torch.device(args.device); cache=torch.load(args.cache,map_location='cpu',weights_only=False); model=RefineCascade(args.variant).to(dev)
    gpu_count=torch.cuda.device_count() if dev.type=='cuda' else 0
    if gpu_count > 1: model=torch.nn.DataParallel(model)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr); best=-1
    def batch(records):
        h=min(r['noisy'].shape[-2] for r in records); w=min(r['noisy'].shape[-1] for r in records)
        return [torch.cat([r[k][...,:h,:w] for r in records],0).to(dev) for k in ('noisy','coarse','clean')]
    for epoch in range(args.epochs):
        model.train()
        train_records=cache['train'][:];
        for start in range(0,len(train_records),args.batch_size):
            raw,coarse,clean=batch(train_records[start:start+args.batch_size]); loss=refine_loss(model(raw,coarse),coarse,clean); opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); vals=[]
        with torch.no_grad():
            for r in cache['test']:
                vals.append(psnr(model(r['noisy'].to(dev),r['coarse'].to(dev))['final'],r['clean'].to(dev)))
        score=sum(vals)/len(vals); print(f'epoch {epoch+1}/{args.epochs}: BSD68 PSNR {score:.4f} dB');
        if score>best:
            best=score; Path(args.out).mkdir(parents=True,exist_ok=True); state=model.module.state_dict() if isinstance(model,torch.nn.DataParallel) else model.state_dict(); torch.save(state,Path(args.out)/f'{cache["denoiser"].lower()}_sigma25_refine_{args.variant}_best.pth'); (Path(args.out)/'metrics.json').write_text(json.dumps({'denoiser':cache['denoiser'],'variant':args.variant,'sigma':cache['sigma'],'baseline_psnr':sum(psnr(r['coarse'],r['clean']) for r in cache['test'])/len(cache['test']),'refine_psnr':score,'gpu_count':gpu_count},indent=2))
if __name__=='__main__': main()
