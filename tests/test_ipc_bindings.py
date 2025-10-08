
import pytest
from google.protobuf.runtime_version import VersionError

from sundew_core_sdk.ipc.bindings import (
    load_grpc_module,
    load_proto_module,
)


def _load_or_skip(loader):
    try:
        module = loader()
    except VersionError as exc:
        pytest.skip(f"protobuf runtime mismatch: {exc}")
    return module


def test_proto_module_loads_real_bindings():
    module = _load_or_skip(load_proto_module)
    assert module is not None
    assert hasattr(module, "ScoreEvent")


def test_grpc_module_loads_real_bindings():
    module = _load_or_skip(load_grpc_module)
    assert module is not None
    assert hasattr(module, "SundewGateServicer")
