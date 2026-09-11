import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
import torch
from refine_bridge_common import add_awgn, RefineCascade

def test_noise_and_cascade_shape():
    clean=torch.zeros(1,1,8,10); noisy=add_awgn(clean,25,1); out=RefineCascade('groupnorm')(noisy,noisy)['final']; assert noisy.shape==clean.shape==out.shape
