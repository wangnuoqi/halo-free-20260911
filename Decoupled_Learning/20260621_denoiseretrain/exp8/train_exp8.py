"""Exp8 ABCD: DnCNN/FFDNet x full-image/patch online joint Exp6E training."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from exp7.train_exp7 import main

if __name__=='__main__': main()
