import types

from distributed_compat import patch_tcpstore_without_libuv


def test_tcpstore_patch_forces_libuv_off():
    calls = []

    def original_tcpstore(*args, **kwargs):
        calls.append((args, kwargs))
        return "store"

    dist_module = types.SimpleNamespace(TCPStore=original_tcpstore)
    rendezvous_module = types.SimpleNamespace(TCPStore=original_tcpstore)

    patch_tcpstore_without_libuv(dist_module, rendezvous_module)

    assert dist_module.TCPStore("host", 29501, 4, False, None) == "store"
    assert calls[-1][1]["use_libuv"] is False

