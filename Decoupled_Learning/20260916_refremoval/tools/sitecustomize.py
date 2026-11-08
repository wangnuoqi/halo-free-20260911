"""Process-wide compatibility hooks for the Windows training environment."""

from __future__ import annotations

import os


if os.environ.get("DISABLE_TORCH_LIBUV") == "1":
    try:
        from distributed_compat import configure_local_rendezvous

        configure_local_rendezvous()
    except Exception as _error:
        print(f"[sitecustomize] Failed to disable torch libuv: {_error}")
