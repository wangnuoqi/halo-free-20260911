import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
import torch
from refine_bridge_common import RefineCascade

def test_both_refine_variants_preserve_shape_and_trainable_params():
    raw=torch.rand(1,1,16,18); coarse=torch.rand(1,1,16,18)
    for variant in ('groupnorm','original'):
        m=RefineCascade(variant); out=m(raw,coarse); assert out['final'].shape==raw.shape
        out['final'].mean().backward(); assert any(p.grad is not None for p in m.parameters())
