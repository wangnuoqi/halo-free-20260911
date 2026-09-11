import importlib.util
from pathlib import Path
import math
import pytest


def load_script():
    path = Path(__file__).with_name("run_inference_compare.py")
    spec = importlib.util.spec_from_file_location("compare_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_baseline_definition_load_does_not_construct_dataset():
    root = Path(__file__).resolve().parents[1]
    module = load_script()
    definition = module.load_definitions(
        root / "models" / "baseline_original" / "refine_definition.py",
        "baseline_definition_test",
    )
    assert hasattr(definition, "CascadeRefine3Stage")


def test_resolve_test_dirs_selects_matching_parameterized_20260813_layout(tmp_path):
    module = load_script()
    for name in (
        "testinput-l1p000-a1p600-200",
        "testlabel-l1p000-a1p600-200",
        "testinput-l1p000-a1p600-200_downsample",
    ):
        (tmp_path / name).mkdir()
    # A non-test subset must not be selected.
    (tmp_path / "testinput-l1p000-a1p600").mkdir()
    origin, label, lr = module.resolve_test_dirs(tmp_path)
    assert origin.name == "testinput-l1p000-a1p600-200"
    assert label.name == "testlabel-l1p000-a1p600-200"
    assert lr.name == "testinput-l1p000-a1p600-200_downsample"


def test_sample_key_matches_20260813_input_and_wls_label_names():
    module = load_script()
    assert module.sample_key(Path("9C4A0004-db1d4a14f3_cropped-log.tiff")) == "9C4A0004-db1d4a14f3_cropped"
    assert module.sample_key(Path("9C4A0004-db1d4a14f3_cropped-WLS-l1p000-a1p600.tiff")) == "9C4A0004-db1d4a14f3_cropped"


def test_model_config_uses_20260813_checkpoints_for_20260813():
    module = load_script()
    config = module.resolve_model_config(Path("/mnt/d/nuoqi/S2R-HDR/data/20260813"))
    assert "20260813" in str(config["baseline_rf"])
    assert "exp7c_recipe_20260813" in str(config["new_rf"])


def test_model_config_uses_exp7c_checkpoints_for_20260818():
    module = load_script()
    config = module.resolve_model_config(Path("/mnt/d/nuoqi/S2R-HDR/data/20260818"))
    assert "baseline_20260818" in str(config["baseline_rf"])
    assert "exp7c_recipe_20260818" in str(config["new_rf"])


def test_model_config_accepts_test_fei_generated_protocols():
    module = load_script()
    config_13 = module.resolve_model_config(Path("/mnt/d/nuoqi/S2R-HDR/data/test_fei/generated_20260813"))
    config_18 = module.resolve_model_config(Path("/mnt/d/nuoqi/S2R-HDR/data/test_fei/generated_20260818"))
    config_flat = module.resolve_model_config(Path("/mnt/d/nuoqi/S2R-HDR/data/test_fei/testinput"))
    config_root = module.resolve_model_config(Path("/mnt/d/nuoqi/S2R-HDR/data/test_fei"))
    assert "20260813" in str(config_13["new_rf"])
    assert "20260818" in str(config_18["new_rf"])
    assert "exp7c_recipe_mixed" in str(config_flat["new_rf"])
    assert "exp7c_recipe_mixed" in str(config_root["new_rf"])


def test_test_vivo_label_protocol_selects_matching_models():
    module = load_script()
    c13 = module.resolve_model_config(Path("/mnt/d/nuoqi/S2R-HDR/data/test_vivo"), Path("testlabel-l1p000-a1p600"))
    c18 = module.resolve_model_config(Path("/mnt/d/nuoqi/S2R-HDR/data/test_vivo"), Path("testlabel-l0p500-a1p400"))
    cbase = module.resolve_model_config(Path("/mnt/d/nuoqi/S2R-HDR/data/test_vivo"), Path("testlabel-l0p250-a1p200"))
    assert "baseline_20260813" in str(c13["baseline_rf"])
    assert "baseline_20260818" in str(c18["baseline_rf"])
    assert "baseline_original" in str(cbase["baseline_rf"])


def test_parameter_tag_selects_test_vivo_directories(tmp_path):
    module = load_script()
    for name in ("testinput-l1p000-a1p600", "testlabel-l1p000-a1p600",
                 "testinput-l1p000-a1p600_downsample"):
        (tmp_path / name).mkdir()
    origin, label, lr = module.resolve_test_dirs(tmp_path, "l1p000-a1p600")
    assert origin.name == "testinput-l1p000-a1p600"
    assert label.name == "testlabel-l1p000-a1p600"
    assert lr.name == "testinput-l1p000-a1p600_downsample"


def test_metric_summary_reports_training_style_mean_per_image_psnr():
    module = load_script()
    mse = [1e-4, 1e-2]
    summary = module.summarize_metrics(mse)
    assert summary["mean_mse"] == 0.00505
    assert summary["psnr_from_mean_mse"] == pytest.approx(10 * math.log10(1 / 0.00505))
    assert summary["mean_per_image_psnr"] == pytest.approx((40.0 + 20.0) / 2)
