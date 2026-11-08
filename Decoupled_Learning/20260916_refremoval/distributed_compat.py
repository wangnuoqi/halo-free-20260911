"""Windows distributed-runtime compatibility helpers."""

from __future__ import annotations

import importlib
import os
from typing import Any


def patch_tcpstore_without_libuv(
    dist_module: Any | None = None,
    rendezvous_module: Any | None = None,
) -> None:
    """Force TCPStore to use the non-libuv implementation on Windows."""
    if os.name != "nt" and dist_module is None:
        return

    if dist_module is None:
        import torch.distributed as dist_module
    if rendezvous_module is None:
        rendezvous_module = importlib.import_module("torch.distributed.rendezvous")

    original = dist_module.TCPStore

    def tcp_store_without_libuv(*args: Any, **kwargs: Any) -> Any:
        kwargs["use_libuv"] = False
        return original(*args, **kwargs)

    dist_module.TCPStore = tcp_store_without_libuv
    if hasattr(rendezvous_module, "TCPStore"):
        rendezvous_module.TCPStore = tcp_store_without_libuv


def configure_local_rendezvous() -> None:
    """Avoid stale hostname/libuv settings inherited by Windows workers."""
    if os.name == "nt":
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29501")
        os.environ["USE_LIBUV"] = "0"
        os.environ["TORCH_USE_LIBUV"] = "0"
        patch_tcpstore_without_libuv()

