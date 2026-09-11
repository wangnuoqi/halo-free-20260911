"""Windows/wintrain launcher for the four frozen-denoiser refine jobs."""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent

def run(cmd, env):
    print('[launcher] START', ' '.join(map(str, cmd)), flush=True)
    subprocess.run([str(x) for x in cmd], env=env, check=True)
    print('[launcher] DONE', cmd[-1] if cmd else '', flush=True)

def valid_cache(path, expected=''):
    if not path.is_file() or path.stat().st_size == 0:
        return False
    import torch
    try:
        x=torch.load(path, map_location='cpu', weights_only=False)
        return bool(x['train']) and bool(x['test']) and float(x['sigma']) == 25.0 and (not expected or x['denoiser'] == expected)
    except Exception:
        return False

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root', default=r'D:\nuoqi\halo-free\data_bsds')
    ap.add_argument('--run-root', default=str(HERE / 'refine_runs' / 'bsds400_bsd68_sigma25'))
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--jobs', choices=['all', 'dncnn', 'ffdnet'], default='all')
    args=ap.parse_args()
    data=Path(args.data_root); run_root=Path(args.run_root); run_root.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']='0,1,2,3'; env['PYTHONUNBUFFERED']='1'
    dn_cache=run_root/'dncnn_sigma25_cache.pt'; ff_cache=run_root/'ffdnet_sigma25_cache.pt'
    dn_gen=HERE/'DnCNN'/'refine_bridge'/'generate_cache.py'; ff_gen=HERE/'FFDNet'/'refine_bridge'/'generate_cache.py'
    if not valid_cache(dn_cache, 'DnCNN'):
        dn_cache.unlink(missing_ok=True)
        run([sys.executable,'-u',dn_gen,'--train-root',data/'BSDS 400','--test-root',data/'BSD68','--out',dn_cache,'--sigma','25','--device','cpu'],env)
    if not valid_cache(ff_cache, 'FFDNet-KAIR-official-PyTorch'):
        ff_cache.unlink(missing_ok=True)
        run([sys.executable,'-u',ff_gen,'--train-root',data/'BSDS 400','--test-root',data/'BSD68','--out',ff_cache,'--sigma','25','--device','cpu'],env)
    jobs=[]
    if args.jobs in ('all','dncnn'):
        jobs += [('DnCNN',dn_cache,'groupnorm',HERE/'DnCNN'/'refine_bridge'/'train_refine.py'),('DnCNN',dn_cache,'original',HERE/'DnCNN'/'refine_bridge'/'train_refine.py')]
    if args.jobs in ('all','ffdnet'):
        jobs += [('FFDNet',ff_cache,'groupnorm',HERE/'FFDNet'/'refine_bridge'/'train_refine.py'),('FFDNet',ff_cache,'original',HERE/'FFDNet'/'refine_bridge'/'train_refine.py')]
    for denoiser,cache,variant,trainer in jobs:
        out=run_root/f'{denoiser}_sigma25_refine_{variant}'
        run([sys.executable,'-u',trainer,'--cache',cache,'--variant',variant,'--epochs',str(args.epochs),'--batch-size',str(args.batch_size),'--device',args.device,'--out',out],env)
    print('[launcher] ALL FOUR EXPERIMENTS COMPLETED',flush=True)

if __name__=='__main__': main()
