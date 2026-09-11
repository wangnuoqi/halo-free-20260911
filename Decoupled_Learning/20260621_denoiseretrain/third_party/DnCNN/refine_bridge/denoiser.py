from pathlib import Path
import importlib.util, os, sys, torch
import torch.nn as nn

def _load_original_dncnn_class():
    """Import the exact class from the shipped training source."""
    source = Path(__file__).parents[1] / 'TrainingCodes' / 'dncnn_pytorch' / 'main_train.py'
    data_dir = str(source.parent)
    if data_dir not in sys.path:
        sys.path.insert(0, data_dir)
    spec = importlib.util.spec_from_file_location('dncnn_original_main_train', source)
    module = importlib.util.module_from_spec(spec)
    old_argv = sys.argv
    old_mkdir = os.mkdir
    try:
        sys.argv = [str(source)]
        # The original module creates its training output directory at import
        # time; suppress that side effect because this bridge only needs DnCNN.
        os.mkdir = lambda *args, **kwargs: None
        spec.loader.exec_module(module)
    finally:
        sys.argv = old_argv
        os.mkdir = old_mkdir
    return module.DnCNN

DnCNN = _load_original_dncnn_class()

def load_dncnn(checkpoint, device='cpu'):
    model=DnCNN().to(device)
    old=getattr(sys.modules['__main__'],'DnCNN',None); sys.modules['__main__'].DnCNN=DnCNN
    try: obj=torch.load(checkpoint,map_location=device,weights_only=False)
    finally:
        if old is not None: sys.modules['__main__'].DnCNN=old
    if isinstance(obj,nn.Module): model=obj.to(device)
    else:
        state=obj.get('state_dict',obj); model.load_state_dict(state,strict=True)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model
