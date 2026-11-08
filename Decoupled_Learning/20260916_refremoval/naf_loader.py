"""Load NAFNet architecture without triggering the BasicSR model registry."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any


def _load_module(module_name: str, module_path: Path) -> Any:
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_package(name: str, package_path: Path) -> None:
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(package_path)]
        sys.modules[name] = package


def load_nafnet_class() -> Any:
    """Return NAFNet while bypassing `basicsr.models.__init__` imports."""
    root = Path(__file__).resolve().parent
    basicsr_dir = root / "basicsr"
    models_dir = basicsr_dir / "models"
    archs_dir = root / "basicsr" / "models" / "archs"

    _ensure_package("basicsr", basicsr_dir)
    _ensure_package("basicsr.models", models_dir)
    _ensure_package("basicsr.models.archs", archs_dir)

    _load_module("basicsr.models.archs.arch_util", archs_dir / "arch_util.py")
    _load_module("basicsr.models.archs.local_arch", archs_dir / "local_arch.py")
    module = _load_module("nafnet_arch_local", archs_dir / "NAFNet_arch.py")
    return module.NAFNet
