"""Registry for hardware-specific adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Type, TypeVar


T = TypeVar("T", bound="HardwareAdapter")


@dataclass
class HardwareAdapter:
    name: str
    board: str

    def flash(self) -> None:
        raise NotImplementedError

    def monitor(self) -> None:
        raise NotImplementedError


@dataclass
class HardwareRegistry:
    adapters: Dict[str, Type[HardwareAdapter]] = field(default_factory=dict)

    def register(self, adapter: Type[T]) -> Type[T]:
        self.adapters[adapter.__name__] = adapter
        return adapter

    def get(self, name: str) -> Type[HardwareAdapter]:
        return self.adapters[name]


__all__ = ["HardwareAdapter", "HardwareRegistry"]
