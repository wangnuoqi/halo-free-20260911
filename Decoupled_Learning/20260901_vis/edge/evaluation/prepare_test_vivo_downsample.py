"""Create LR TIFFs for the three parameterized test_vivo input sets."""
from pathlib import Path
import sys

PIPELINE = Path(__file__).resolve().parents[3] / "total-pipeline-v1"
sys.path.insert(0, str(PIPELINE))
from downsample_hdr import process_directory  # noqa: E402

ROOT = Path("/mnt/d/nuoqi/S2R-HDR/data/test_vivo")
PARAMS = ("l0p250-a1p200", "l0p500-a1p400", "l1p000-a1p600")

if __name__ == "__main__":
    for tag in PARAMS:
        source = ROOT / f"testinput-{tag}"
        target = ROOT / f"testinput-{tag}_downsample"
        if not source.is_dir():
            raise FileNotFoundError(source)
        print(f"Downsampling: {source} -> {target}")
        process_directory(source, target)
