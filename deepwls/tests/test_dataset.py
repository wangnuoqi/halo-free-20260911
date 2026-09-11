import tempfile
import unittest
from pathlib import Path

from PIL import Image

from util.dataset import PairedImageDataset, build_paired_samples


def write_tiff(path: Path, size=(32, 32), value=0):
    image = Image.new("F", size, color=float(value))
    image.save(path)


class BuildPairedSamplesTest(unittest.TestCase):
    def test_matches_train_input_to_train_label(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "traininput"
            label_dir = root / "trainlabel"
            input_dir.mkdir()
            label_dir.mkdir()

            write_tiff(input_dir / "0001-input.tiff", value=1)
            write_tiff(label_dir / "0001-WLS-025000.tiff", value=2)

            samples = build_paired_samples(input_dir, label_dir, split="train")

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0][0].name, "0001-input.tiff")
            self.assertEqual(samples[0][1].name, "0001-WLS-025000.tiff")

    def test_matches_test_input_to_test_label(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "testinput"
            label_dir = root / "testlabel"
            input_dir.mkdir()
            label_dir.mkdir()

            write_tiff(input_dir / "0001-input.tiff", value=1)
            write_tiff(label_dir / "0001.tiff-WLS-025000.tiff", value=2)

            samples = build_paired_samples(input_dir, label_dir, split="test")

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0][0].name, "0001-input.tiff")
            self.assertEqual(samples[0][1].name, "0001.tiff-WLS-025000.tiff")

    def test_matches_test_input_with_embedded_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "testinput"
            label_dir = root / "testlabel"
            input_dir.mkdir()
            label_dir.mkdir()

            write_tiff(input_dir / "0140.tiff-input.tif", value=1)
            write_tiff(label_dir / "0140.tiff-WLS-025000.tiff", value=2)

            samples = build_paired_samples(input_dir, label_dir, split="test")

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0][2], "0140")
            self.assertEqual(samples[0][1].name, "0140.tiff-WLS-025000.tiff")

    def test_matches_train_label_that_keeps_input_suffix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "traininput"
            label_dir = root / "trainlabel"
            input_dir.mkdir()
            label_dir.mkdir()

            write_tiff(input_dir / "AG8A9036-6ff35fa9a6-input.tiff", value=1)
            write_tiff(label_dir / "AG8A9036-6ff35fa9a6-input-WLS-025000.tiff", value=2)

            samples = build_paired_samples(input_dir, label_dir, split="train")

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0][1].name, "AG8A9036-6ff35fa9a6-input-WLS-025000.tiff")

    def test_raises_when_label_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "traininput"
            label_dir = root / "trainlabel"
            input_dir.mkdir()
            label_dir.mkdir()

            write_tiff(input_dir / "0001-input.tiff", value=1)

            with self.assertRaisesRegex(FileNotFoundError, "0001"):
                build_paired_samples(input_dir, label_dir, split="train")


class PairedImageDatasetTest(unittest.TestCase):
    def test_returns_paired_random_crops(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "traininput"
            label_dir = root / "trainlabel"
            input_dir.mkdir()
            label_dir.mkdir()

            write_tiff(input_dir / "0001-input.tiff", size=(64, 64), value=1)
            write_tiff(label_dir / "0001-WLS-025000.tiff", size=(64, 64), value=2)

            dataset = PairedImageDataset(input_dir, label_dir, split="train", patch_size=32)
            input_tensor, label_tensor, name = dataset[0]

            self.assertEqual(tuple(input_tensor.shape), (3, 32, 32))
            self.assertEqual(tuple(label_tensor.shape), (3, 32, 32))
            self.assertEqual(name, "0001")


if __name__ == "__main__":
    unittest.main()
