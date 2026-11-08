import shlex

import pytest

from expe.model import EXPE_VARIANTS
from expe.wintrain_command import command, main


@pytest.mark.parametrize("variant", sorted(EXPE_VARIANTS))
def test_every_variant_has_isolated_scratch_command(variant):
    generated = command(variant, "scratch", "seed42", 42, 5000, "base", None, None)
    executable, payload = shlex.split(generated)
    assert executable == "wintrain"
    assert f"--variant {variant} " in payload
    assert rf"\runs\scratch\{variant}\seed42 " in payload
    assert "--init-lr-checkpoint" not in payload
    assert "--resume" not in payload


@pytest.mark.parametrize("variant", sorted(set(EXPE_VARIANTS) - {"viv_gf_e1"}))
def test_paired_ablation_uses_one_frozen_e1_source(variant):
    payload = shlex.split(command(variant, "paired", "seed42", 42, 5000,
                                  "base", None, None))[1]
    assert rf"\runs\paired\{variant}\base\seed42 " in payload
    assert r"\runs\scratch\viv_gf_e1\seed42\best.pth --freeze-lr" in payload


def test_resume_keeps_run_location_and_initialization():
    initial = shlex.split(command("viv_gf_c2", "paired", "seed42", 42,
                                    5000, "base", None, None))[1]
    resumed = shlex.split(command("viv_gf_c2", "paired", "seed42", 42,
                                    60000, "base", None, None, resume=True))[1]
    assert initial.replace("--stop-after 5000", "--stop-after 60000") + " --resume" == resumed


def test_default_command_runs_to_total_iteration(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["wintrain_command.py", "--phase", "scratch",
                                  "--variant", "viv_gf_e1"])
    main()
    assert "--stop-after 60000" in capsys.readouterr().out
