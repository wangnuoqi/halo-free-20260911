"""Load the unmodified single-operator model classes from the official script."""

import ast
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_official_model(script_path):
    """Load only Model and Resize definitions, avoiding the script's training side effects."""
    source_path = Path(script_path)
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    wanted = {"Model", "Resize"}
    definitions = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in wanted
    ]
    found = {node.name for node in definitions}
    if found != wanted:
        raise RuntimeError(f"Expected official classes {sorted(wanted)} in {source_path}; found {sorted(found)}")
    module = ast.Module(body=definitions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch, "nn": nn, "F": F, "math": math}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["Model"]
