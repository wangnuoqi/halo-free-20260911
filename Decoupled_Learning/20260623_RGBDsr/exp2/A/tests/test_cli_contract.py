from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "train_convguided_base.py"
EVALUATOR = Path(__file__).resolve().parents[1] / "test_convguided_base.py"


def test_exp2_a_has_independent_cli_paths_and_macro_evaluator():
    train_source = SCRIPT.read_text(encoding="utf-8")
    test_source = EVALUATOR.read_text(encoding="utf-8")

    assert "--train-root" in train_source
    assert "--test-root" in train_source
    assert "exp2_A_guide_smooth_x{args.scale}" in train_source
    assert 'np.mean([row["rmse_mm"] for row in rows])' in test_source
