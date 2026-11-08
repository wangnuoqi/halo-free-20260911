"""Load only the VIV-LR predictor from an Exp E1 checkpoint."""

from pathlib import Path

import torch


def initialize_lr(model, checkpoint_path: Path, freeze: bool) -> None:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if state.get("protocol", {}).get("variant") != "viv_gf_e1":
        raise ValueError("LR initialization requires a viv_gf_e1 checkpoint")
    lr_state = {name.removeprefix("lr."): value
                for name, value in state["model"].items() if name.startswith("lr.")}
    model.lr.load_state_dict(lr_state, strict=True)
    model.lr_frozen = freeze
    for parameter in model.lr.parameters():
        parameter.requires_grad_(not freeze)
    if freeze:
        model.lr.eval()


def initialize_model(model, checkpoint_path: Path, variant: str) -> None:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if state.get("protocol", {}).get("variant") != variant:
        raise ValueError("Full-model initialization requires the same Exp E variant")
    model.load_state_dict(state["model"], strict=True)
    model.lr_frozen = False
    for parameter in model.lr.parameters():
        parameter.requires_grad_(True)
