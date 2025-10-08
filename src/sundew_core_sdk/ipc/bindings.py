"""Helpers to load generated protobuf bindings."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Optional


DEFAULT_MODULE = 'sundew_ipc_v1_pb2'
DEFAULT_GRPC_MODULE = 'sundew_ipc_v1_pb2_grpc'


def load_proto_module(module: str = DEFAULT_MODULE) -> Optional[ModuleType]:
    """Attempt to import the generated protobuf module."""

    try:
        return importlib.import_module(module)
    except ModuleNotFoundError:
        return None


def load_grpc_module(module: str = DEFAULT_GRPC_MODULE) -> Optional[ModuleType]:
    """Attempt to import the generated gRPC module."""

    try:
        return importlib.import_module(module)
    except ModuleNotFoundError:
        return None


__all__ = ['load_proto_module', 'load_grpc_module', 'DEFAULT_MODULE', 'DEFAULT_GRPC_MODULE']
