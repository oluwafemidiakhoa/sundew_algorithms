"""IPC utilities for firmware communication."""

from .bindings import load_proto_module
from .shim import SundewGateShim

__all__ = ['load_proto_module', 'SundewGateShim']
