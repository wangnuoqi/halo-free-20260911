"""Probe the refinement ceiling when the low-resolution prediction is GT-derived."""
import argparse, csv, json, math
from pathlib import Path
import cv2, numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/mnt/d/ResearchCode/S2R-HDR/data/20260818')
EXP = ROOT / 'experiments/exp7/exp7C_20260818_joint_fidelity/models/exp7C_20260818_joint_fidelity'
SOURCE = ROOT / 'experiments/exp7C_joint_fidelity_20260818.py'

def defs(path):
    s=path.read_text(); marker='\ntrain_dataset = GuidedFilterDataset('
    ns={'__name__':'gt_lr_probe','__file__':str(path)}
    exec(compile(s.split(marker,1)[0],str(path),'exec'),ns); return ns
def load(m,p,d):
    x=torch.load(p,map_location=d,weights_only=False)
    if isinstance(x,dict) and 'model_state_dict' in x:x=x['model_state_dict']
    if all(k.startswith('module.') for k in x):x={k[7:]:v for k,v in x.items()}
    m.load_state_dict(x,strict=True)
def even(x): return x[:x.shape[0]-x.shape[0]%2,:x.shape[1]-x.shape[1]%2]
def metric(p,y):
    mse=float(np.mean((p-y)**2)); return mse,10*math.log10(1/max(mse,1e-12))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',type=Path,default=DATA); ap.add_argument('--experiment-root',type=Path,default=EXP); ap.add_argument('--source',type=Path,default=SOURCE); ap.add_argument('--output-dir',type=Path,default=ROOT/'baseline/evaluation/gt_lr_coarse_probe'); ap.add_argument('--device',default='cpu'); args=ap.parse_args()
    d=torch.device(args.device); ns=defs(args.source)
    conv=ns['DeepGuidedFilter'](radius=1,eps=1e-8).to(d); ref=ns['CascadeRefine3Stage'](base_channels=32,a_scale=.2,b_scale=.2,use_delta_as_input=True).to(d)
    load(conv,args.experiment_root/'convguided_joint_best_20260818_l0p500_a1p400.pth',d); load(ref,args.experiment_root/'rf_3stage_best_20260818_l0p500_a1p400.pth',d); conv.eval();ref.eval()
    root=args.data_root; lrs=sorted((root/'testinput-l0p500-a1p400-200_downsample').iterdir()); ops=sorted((root/'testinput-l0p500-a1p400-200').iterdir()); ys=sorted((root/'testlabel-l0p500-a1p400-200').iterdir())
    rows=[]; keys=['origin','gt_lr_bilinear','pred_lr_guided','pred_lr_refined','gt_lr_guided','gt_lr_refined']
    vals={k:[] for k in keys}
    with torch.inference_mode():
      for n,(lp,op,yp) in enumerate(zip(lrs,ops,ys),1):
        lr=cv2.imread(str(lp),-1).astype('float32'); x=even(cv2.imread(str(op),-1).astype('float32')); gt=even(cv2.imread(str(yp),-1).astype('float32'))
        h,w=lr.shape; gt_lr=cv2.resize(gt,(w,h),interpolation=cv2.INTER_AREA).astype('float32')
        lr_t=torch.from_numpy(lr)[None,None].to(d); x_t=torch.from_numpy(x)[None,None].to(d); glr_t=torch.from_numpy(gt_lr)[None,None].to(d)
        x_lr=lr_t-.5; x_hr=x_t-.5
        pred_lr=conv.lr(x_lr)
        normal_coarse=conv(x_lr,x_hr)
        # Match DeepGuidedFilter.forward exactly: clamp the guided output before refine.
        normal_direct=conv.gf(x_lr,pred_lr,x_hr).clamp(0,1)
        if not torch.allclose(normal_coarse,normal_direct,atol=1e-6,rtol=1e-5):
          raise RuntimeError('normal guided-filter path does not match conv.forward')
        # conv.lr emits an image-domain [0,1] prediction. Only x_lr/x_hr are
        # centered in the trained ConvGuided call, so keep GT-LR uncentered.
        oracle_coarse=conv.gf(x_lr,glr_t,x_hr).clamp(0,1)
        bil=cv2.resize(gt_lr,(w and gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_LINEAR).astype('float32')
        normal_pred=(ref(x_hr,normal_coarse-.5)['final']+.5).clamp(0,1)
        oracle_pred=(ref(x_hr,oracle_coarse-.5)['final']+.5).clamp(0,1)
        ps={'origin':x,'gt_lr_bilinear':bil,
            'pred_lr_guided':normal_coarse[0,0].cpu().numpy(),
            'pred_lr_refined':normal_pred[0,0].cpu().numpy(),
            'gt_lr_guided':oracle_coarse[0,0].cpu().numpy(),
            'gt_lr_refined':oracle_pred[0,0].cpu().numpy()}
        row={'id':op.stem}
        for k,v in ps.items():
          m,p=metric(v,gt); vals[k].append((m,p)); row[k+'_mse']=m;row[k+'_psnr_db']=p
        rows.append(row)
        if n%25==0 or n==len(lrs):print(f'test {n}/{len(lrs)}',flush=True)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with (args.output_dir/'per_image_metrics.csv').open('w',newline='') as f:
      wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    baseline_psnr=float(np.mean([z[1] for z in vals['pred_lr_refined']]))
    if abs(baseline_psnr-47.11408764719622)>0.02:
      raise RuntimeError(f'baseline path did not reproduce existing 47.114 dB result: {baseline_psnr:.4f} dB')
    out={'protocol':'20260818 test 200, even crop, full image, per-image PSNR mean; GT INTER_AREA-resized to input LR dimensions; only conv.lr(x_lr) replaced by GT-LR in image domain; guided output clamped [0,1], refine receives coarse minus 0.5','warning':'GT-LR is an oracle input unavailable at inference; refine checkpoint trained on predicted coarse, not oracle coarse','test_images':len(rows),'mean_of_per_image_mse':{k:float(np.mean([z[0] for z in v])) for k,v in vals.items()},'mean_of_per_image_psnr_db':{k:float(np.mean([z[1] for z in v])) for k,v in vals.items()}}
    (args.output_dir/'summary.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__':main()
