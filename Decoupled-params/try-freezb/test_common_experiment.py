import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import types
import re


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.SimpleNamespace(IMREAD_UNCHANGED=-1, imread=lambda *args, **kwargs: None)

import common_experiment as ce
import eval_try_freezb as eval_script


class GuidedFilterDatasetDiagnosticsTests(unittest.TestCase):
    def test_getitem_includes_path_when_cv2_read_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            x_lr_dir = root / "x_lr"
            x_hr_dir = root / "x_hr"
            label_dir = root / "label"
            x_lr_dir.mkdir()
            x_hr_dir.mkdir()
            label_dir.mkdir()

            x_lr_path = x_lr_dir / "sample-l0.1-a1.0-h.png"
            x_hr_path = x_hr_dir / "sample-l0.1-a1.0-h.png"
            label_path = label_dir / "sample-l0.1-a1.0-h.png"
            x_lr_path.write_bytes(b"")
            x_hr_path.write_bytes(b"")
            label_path.write_bytes(b"")

            dataset = ce.GuidedFilterDataset(str(x_lr_dir), str(x_hr_dir), str(label_dir))

            with mock.patch.object(ce.cv2, "imread", return_value=None):
                with self.assertRaisesRegex(RuntimeError, re.escape(str(x_lr_path))):
                    dataset[0]


class EvalTryFreezbConfigTests(unittest.TestCase):
    def test_eval_configs_include_v5(self) -> None:
        self.assertIn("v5", eval_script.EXPERIMENT_CONFIGS)
        config = eval_script.EXPERIMENT_CONFIGS["v5"]
        self.assertEqual(config.cond_mode, "mlp8")
        self.assertEqual(config.cond_channels, 8)
        self.assertEqual(config.freeze_mode, "progressive_v5")
        self.assertTrue(config.use_front_adapter)

    def test_parse_args_accepts_v5_variant(self) -> None:
        with mock.patch.object(sys, "argv", ["eval_try_freezb.py", "--variants", "v5"]):
            args = eval_script.parse_args()
        self.assertEqual(args.variants, ["v5"])


if __name__ == "__main__":
    unittest.main()
