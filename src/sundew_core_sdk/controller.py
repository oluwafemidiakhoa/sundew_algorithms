"""Adaptive gate controller facade for embedded clients."""

from __future__ import annotations

from typing import Any, Dict

import random

from .config import SDKConfig
from .telemetry import TelemetryEvent


class AdaptiveGateController:
    """Placeholder facade wrapping Sundew's adaptive gating logic."""

    def __init__(self, config: SDKConfig) -> None:
        self.config = config
        self._native_gate = None  # TODO: integrate with sundew.core.SundewAlgorithm

    def load_native(self) -> None:
        """Instantiate the legacy Sundew algorithm as a stop-gap."""
        try:
            from sundew.core import SundewAlgorithm
            from sundew.config import SundewConfig

            legacy_cfg = SundewConfig()
            if hasattr(legacy_cfg, 'activation_threshold'):
                legacy_cfg.activation_threshold = max(0.0, min(1.0, self.config.target_activation))
            if hasattr(legacy_cfg, 'gate_temperature'):
                legacy_cfg.gate_temperature = self.config.gate_temperature
            if hasattr(legacy_cfg, 'max_energy'):
                legacy_cfg.max_energy = self.config.max_energy
            self._native_gate = SundewAlgorithm(legacy_cfg)
        except Exception as exc:  # pragma: no cover - placeholder path
            raise RuntimeError("Native Sundew algorithm unavailable") from exc

    def decide(self, features: Dict[str, Any]) -> bool:
        """Return True when heavy inference should run."""
        if self._native_gate is None:
            raise RuntimeError("Controller not initialized; call load_native() first")
        result = self._native_gate.process(features)
        activated = result is not None
        if not activated:
            return False
        target = max(0.0, min(1.0, self.config.target_activation))
        if target <= 0.0:
            return False
        if target >= 0.999:
            return True
        return random.random() < target

    def emit_telemetry(self) -> TelemetryEvent:
        """Return minimal telemetry for the host firmware."""
        return TelemetryEvent(
            activation_rate=0.0,
            threshold=0.0,
            energy_level=0.0,
        )


__all__ = ["AdaptiveGateController"]
