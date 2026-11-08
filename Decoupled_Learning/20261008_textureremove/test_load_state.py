import torch

from rtv_training_common import DeepGuidedFilter, load_state


def test_load_state_keeps_module_prefix_for_dataparallel(tmp_path):
    source = torch.nn.DataParallel(DeepGuidedFilter())
    checkpoint = tmp_path / "full.pth"
    torch.save({"model_state_dict": source.state_dict()}, checkpoint)

    target = torch.nn.DataParallel(DeepGuidedFilter())
    load_state(target, checkpoint, torch.device("cpu"))

