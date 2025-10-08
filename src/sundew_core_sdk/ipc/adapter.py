
"""IPC adapter bridging protobuf messages with the Sundew gate controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..controller import AdaptiveGateController
from ..metrics import MetricsTracker
from ..telemetry import TelemetryEvent

try:
    from sundew_ipc_v1_pb2 import GateDecision, ScoreEvent, TelemetryPush
except ModuleNotFoundError:  # pragma: no cover - bindings optional
    GateDecision = ScoreEvent = TelemetryPush = None


@dataclass
class IPCAdapter:
    controller: AdaptiveGateController
    tracker: MetricsTracker

    def handle_score_event(self, event: ScoreEvent | object):
        if ScoreEvent is None:
            raise RuntimeError("ScoreEvent bindings not available")
        features = {kv.key: kv.value for kv in event.features}  # type: ignore[union-attr]
        activated = self.controller.decide(features)
        decision = GateDecision(sequence=event.sequence)  # type: ignore[union-attr]
        decision.should_activate = activated
        decision.confidence = 1.0 if activated else 0.2
        self.tracker.record(activated, features.get("power", 0.0))
        return decision

    def record_telemetry(self, push: TelemetryPush | object) -> TelemetryEvent:
        if TelemetryPush is None:
            raise RuntimeError("TelemetryPush binding not available")
        self.tracker.record(push.activation_rate > 0.0, push.average_power_w)  # type: ignore[union-attr]
        snap = self.tracker.snapshot()
        return TelemetryEvent(
            activation_rate=snap.activation_rate,
            threshold=0.0,
            energy_level=snap.energy_buffer,
        )
