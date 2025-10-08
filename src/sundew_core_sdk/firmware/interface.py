"""Reference firmware interface for Sundew Core SDK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..telemetry import TelemetryEvent


class FirmwareGateInterface(Protocol):
    """Protocol describing the minimal interface firmware must implement."""

    def initialize(self) -> None:
        ...

    def score_event(self, payload: memoryview) -> None:
        ...

    def next_decision(self) -> bool:
        ...

    def latest_telemetry(self) -> TelemetryEvent:
        ...


@dataclass
class FirmwareStatus:
    ok: bool = True
    detail: str = ""


__all__ = ["FirmwareGateInterface", "FirmwareStatus"]
