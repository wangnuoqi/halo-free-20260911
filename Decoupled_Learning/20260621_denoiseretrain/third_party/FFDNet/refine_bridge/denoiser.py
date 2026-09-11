"""Loader for the official-author KAIR PyTorch FFDNet release."""
from pathlib import Path
import importlib.util, sys, torch

OFFICIAL_ROOT = Path(__file__).parent / 'pytorch_official'
OFFICIAL_CHECKPOINT = OFFICIAL_ROOT / 'models' / 'ffdnet_gray.pth'

def _official_network():
    path = OFFICIAL_ROOT / 'network_ffdnet.py'
    sys.path.insert(0, str(OFFICIAL_ROOT))
    spec = importlib.util.spec_from_file_location('kair_official_ffdnet', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FFDNet

FFDNetGray = _official_network()

def load_ffdnet(checkpoint=OFFICIAL_CHECKPOINT, device='cpu'):
    model=FFDNetGray(in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R')
    payload=torch.load(checkpoint, map_location='cpu', weights_only=True)
    state=payload.get('model_state_dict', payload.get('state_dict', payload))
    model.load_state_dict(state, strict=True)
    model=model.to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model
