"""Exp9 uses the shared Exp7/Exp8 trainer with explicit coarse-loss ablations."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))
from exp7.train_exp7 import main


if __name__ == '__main__':
    main()
