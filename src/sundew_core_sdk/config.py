"""Configuration helpers for the Sundew Core SDK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SDKConfig:
    """Minimal placeholder configuration for Phase 1.

    Reuses defaults from the legacy SundewConfig until the dedicated SDK
    surface is finalized.
    """

    target_activation: float = 0.22
    gate_temperature: float = 0.08
    max_energy: float = 100.0
    firmware_endpoint: Optional[str] = None


__all__ = ["SDKConfig"]
