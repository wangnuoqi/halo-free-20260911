import importlib.util
from pathlib import Path


def _launcher_module():
    path = Path(__file__).parents[1] / 'run_exp7.py'
    spec = importlib.util.spec_from_file_location('exp7_launcher', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exp7_has_experiment_local_bases_for_all_denoisers(tmp_path):
    launcher = _launcher_module()
    bases = launcher.base_specs(tmp_path / 'models', {'dncnn': Path('dn-source.pth'), 'ffdnet': Path('ff-source.pth')})

    assert set(bases) == {'convguided', 'dncnn', 'ffdnet'}
    assert bases['dncnn'][0] == tmp_path / 'models' / 'dncnn_base_full' / 'dncnn_best.pth'
    assert bases['ffdnet'][1] == Path('ff-source.pth')


def test_prepare_output_preserves_completed_run_and_removes_incomplete_run(tmp_path):
    launcher = _launcher_module()
    completed = tmp_path / 'completed'
    completed.mkdir()
    (completed / 'training_complete.json').write_text('{}')
    assert launcher.prepare_output(completed) is False

    incomplete = tmp_path / 'incomplete'
    incomplete.mkdir()
    (incomplete / 'manifest.json').write_text('{}')
    assert launcher.prepare_output(incomplete) is True
    assert not incomplete.exists()


def test_exp7_launcher_offers_all_denoiser_rf_batches():
    launcher = _launcher_module()
    source = (Path(__file__).parents[1] / 'run_exp7.py').read_text()
    assert "'frozen-rf'" in source
    assert "'joint-rf'" in source
