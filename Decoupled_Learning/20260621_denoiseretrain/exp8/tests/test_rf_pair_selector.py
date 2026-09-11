import importlib.util
from pathlib import Path


def test_pair_selector_runs_both_modes_for_one_denoiser():
    path = Path(__file__).parents[1] / 'run_exp8.py'
    spec = importlib.util.spec_from_file_location('exp8_launcher', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.select_modes('dncnn-rf', 'dncnn') == ('frozen', 'joint')
    assert module.select_modes('ffdnet-rf', 'dncnn') == ()
