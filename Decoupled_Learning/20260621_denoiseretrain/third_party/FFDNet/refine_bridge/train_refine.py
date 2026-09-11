from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[2]))
from DnCNN.refine_bridge.train_refine import main
if __name__=='__main__': main()
