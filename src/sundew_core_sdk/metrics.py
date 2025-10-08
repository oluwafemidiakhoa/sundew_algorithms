
"""Metrics helpers for the Sundew Core SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MetricsSnapshot:
    activation_rate: float
    avg_power_w: float
    energy_buffer: float
    samples: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "activation_rate": self.activation_rate,
            "avg_power_w": self.avg_power_w,
            "energy_buffer": self.energy_buffer,
            "samples": float(self.samples),
        }


@dataclass
class MetricsTracker:
    """Collect rolling metrics for activation and power observations."""

    window: int = 2048
    activations: List[int] = field(default_factory=list)
    power_samples: List[float] = field(default_factory=list)
    energy_buffer: float = 0.0

    def record(self, activated: bool, power_w: float, energy: float | None = None) -> None:
        self.activations.append(1 if activated else 0)
        self.power_samples.append(float(power_w))
        if len(self.activations) > self.window:
            self.activations.pop(0)
        if len(self.power_samples) > self.window:
            self.power_samples.pop(0)
        if energy is not None:
            self.energy_buffer = float(energy)

    def snapshot(self) -> MetricsSnapshot:
        samples = len(self.activations)
        activation_rate = sum(self.activations) / samples if samples else 0.0
        avg_power = sum(self.power_samples) / samples if samples else 0.0
        return MetricsSnapshot(
            activation_rate=activation_rate,
            avg_power_w=avg_power,
            energy_buffer=self.energy_buffer,
            samples=samples,
        )

    def as_dict(self) -> Dict[str, float]:
        return self.snapshot().as_dict()

    def to_json(self) -> Dict[str, float]:
        return self.as_dict()


__all__ = ["MetricsTracker", "MetricsSnapshot"]
