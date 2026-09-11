import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1])); from DnCNN.refine_bridge.denoiser import load_dncnn
import torch
def test_dncnn_load_and_freeze():
 p=Path(__file__).parents[1]/'DnCNN/TrainingCodes/dncnn_pytorch/models/DnCNN_sigma25/model.pth'; m=load_dncnn(p); y=m(torch.zeros(1,1,8,8)); assert y.shape==(1,1,8,8) and not any(x.requires_grad for x in m.parameters())
