"""Create half-resolution float32 TIFF inputs for all test_fei protocols."""
from pathlib import Path
import sys

# .../Decoupled_Learning/20260901_vis/edge/evaluation -> Decoupled_Learning
PIPELINE = Path(__file__).resolve().parents[3] / "total-pipeline-v1"
sys.path.insert(0, str(PIPELINE))
from downsample_hdr import process_directory  # noqa: E402


ROOT = Path("/mnt/d/nuoqi/S2R-HDR/data/test_fei")
JOBS = (
    (ROOT / "generated_20260813" / "testinput-l1p000-a1p600",
     ROOT / "generated_20260813" / "testinput-l1p000-a1p600_downsample"),
    (ROOT / "generated_20260818" / "testinput-l0p500-a1p400",
     ROOT / "generated_20260818" / "testinput-l0p500-a1p400_downsample"),
    (ROOT / "testinput", ROOT / "testinput_downsample"),
)


if __name__ == "__main__":
    for source, target in JOBS:
        if not source.is_dir():
            raise FileNotFoundError(source)
        print(f"Downsampling: {source} -> {target}")
        process_directory(source, target)
