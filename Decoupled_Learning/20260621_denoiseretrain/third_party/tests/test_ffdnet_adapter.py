import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1])); from FFDNet.refine_bridge.denoiser import load_ffdnet
import torch
def test_ffdnet_load_and_shape():
 p=Path(__file__).parents[1]/'FFDNet/refine_bridge/pytorch_official/models/ffdnet_gray.pth'; m=load_ffdnet(p); y=m(torch.zeros(1,1,8,10),torch.full((1,1,1,1),25/255.)); assert y.shape==(1,1,8,10) and not any(x.requires_grad for x in m.parameters())
