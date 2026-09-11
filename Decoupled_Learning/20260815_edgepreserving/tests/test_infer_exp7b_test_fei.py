import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'infer_exp7b_test_fei.py'
SPEC = importlib.util.spec_from_file_location('infer_exp7b_test_fei', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
format_metrics = MODULE.format_metrics
match_test_triplets = MODULE.match_test_triplets


def test_match_test_triplets_uses_identical_filename_stems(tmp_path):
    for name in ('testinput', 'testinput_downsample', 'testlabel'):
        (tmp_path / name).mkdir()
    (tmp_path / 'testinput' / 'scene-a.tiff').touch()
    (tmp_path / 'testinput_downsample' / 'scene-a.tiff').touch()
    (tmp_path / 'testlabel' / 'scene-a-WLS-025000.tiff').touch()

    triplets = match_test_triplets(tmp_path)

    assert triplets == [(
        Path(tmp_path / 'testinput' / 'scene-a.tiff'),
        Path(tmp_path / 'testinput_downsample' / 'scene-a.tiff'),
        Path(tmp_path / 'testlabel' / 'scene-a-WLS-025000.tiff'),
    )]


def test_match_test_triplets_rejects_missing_filename_match(tmp_path):
    for name in ('testinput', 'testinput_downsample', 'testlabel'):
        (tmp_path / name).mkdir()
    (tmp_path / 'testinput' / 'scene-a.tiff').touch()
    (tmp_path / 'testinput_downsample' / 'scene-a.tiff').touch()
    (tmp_path / 'testlabel' / 'other-WLS-025000.tiff').touch()

    with pytest.raises(FileNotFoundError):
        match_test_triplets(tmp_path)


def test_format_metrics_contains_only_rf_final_metrics():
    assert format_metrics(0.125, 9.0309) == 'MSE: 0.12500000\nPSNR: 9.0309 dB\n'
