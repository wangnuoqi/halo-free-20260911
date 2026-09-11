import pytest
import sys
from pathlib import Path

torch = pytest.importorskip('torch')

sys.path.insert(0, str(Path(__file__).parents[1]))
from convguided_adapter import ConvGuidedGray


def test_fresh_convguided_lr_starts_with_identity_first_convolution():
    model = ConvGuidedGray()
    first = model.lr[0].weight.detach()

    assert torch.isclose(first[0, 0, 1, 1], torch.tensor(1.0))
    assert torch.count_nonzero(first[0, 0]) == 1
    assert torch.count_nonzero(first[1:]) == 0
