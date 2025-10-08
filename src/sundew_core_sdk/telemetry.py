"""Telemetry primitives for exchanging data with firmware layers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TelemetryEvent:
    activation_rate: float
    threshold: float
    energy_level: float


__all__ = ["TelemetryEvent"]
