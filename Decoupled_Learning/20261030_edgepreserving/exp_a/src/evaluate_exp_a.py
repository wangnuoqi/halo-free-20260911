"""Full-image metrics for an Experiment A checkpoint; predictions are not saved."""
import argparse, csv, json, math, sys
from pathlib import Path
import cv2, numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_exp_a import Full, Pairs

def even(x): return x[..., :x.shape[-2]-x.shape[-2]%2, :x.shape[-1]-x.shape[-1]%2]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--data-root',default='/mnt/d/ResearchCode/S2R-HDR/data/20260818'); ap.add_argument('--output-dir',required=True); ap.add_argument('--device',default='cuda:0'); args=ap.parse_args()
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False); p=ck.get('run_protocol',{}); variant=p.get('variant')
    if variant not in {'c0','c1','a0','a1'}: raise ValueError('checkpoint missing Experiment A variant')
    d=torch.device(args.device if args.device=='cpu' or torch.cuda.is_available() else 'cpu'); m=Full(variant in {'c1','a1'},variant=='a1',bounded_viv_output=p.get('lr_output_mode')=='sigmoid_0_1').to(d); m.load_state_dict(ck['model_state'],strict=True); m.eval()
    ds=Pairs(args.data_root,'test'); rows=[]
    with torch.inference_mode():
        for i,(lr,x,y) in enumerate(ds,1):
            lr=lr[None].to(d); x=even(x[None]).to(d); y=even(y[None]).to(d); o=m(lr-.5,x-.5)
            row={'id':ds.hr[i-1].stem}
            for k in ('coarse0','coarse1','coarse2','final'):
                e=(o[k]-y).float(); mse=float(e.square().mean().cpu()); row[k+'_mse']=mse; row[k+'_psnr_db']=10*math.log10(1/max(mse,1e-12))
            rows.append(row)
            if i%25==0 or i==len(ds): print(f'test {i}/{len(ds)}',flush=True)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    with (out/'per_image_metrics.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary={'variant':variant,'checkpoint':str(Path(args.checkpoint).resolve()),'test_images':len(rows),'geometry':'full image, even crop, no prediction files','mean_of_per_image_mse':{k:float(np.mean([r[k+'_mse'] for r in rows])) for k in ('coarse0','coarse1','coarse2','final')},'mean_psnr_db':{k:float(np.mean([r[k+'_psnr_db'] for r in rows])) for k in ('coarse0','coarse1','coarse2','final')}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
