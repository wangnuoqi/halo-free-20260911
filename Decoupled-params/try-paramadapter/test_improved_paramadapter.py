import importlib.util
from pathlib import Path

import torch


SCRIPT_PATH = Path("/mnt/d/nuoqi/halo-free/Decoupled-params/try-paramadapter/train_convguided_base_paramadapter_film.py")


def load_module():
    spec = importlib.util.spec_from_file_location("train_convguided_base_paramadapter_film", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_improved_adapter_defaults_and_forward():
    module = load_module()
    model = module.DeepGuidedFilterAdapterFiLM(radius=1)

    assert abs(model.adapter.scale - 0.5) < 1e-8
    assert model.adapter.hidden == 64
    assert isinstance(model.adapter.proj, torch.nn.Conv2d)
    assert model.adapter.proj.kernel_size == (1, 1)

    x_lr = torch.randn(2, 1, 32, 32)
    x_hr = torch.randn(2, 1, 64, 64)
    params = torch.tensor([[0.1, 1.0], [1.0, 1.4]], dtype=torch.float32)
    out = model(x_lr, x_hr, params)

    assert out.shape == (2, 1, 64, 64)


if __name__ == "__main__":
    test_improved_adapter_defaults_and_forward()
    print("ok")
