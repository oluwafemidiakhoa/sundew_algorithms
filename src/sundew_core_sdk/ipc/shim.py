"""C shim stubs for integrating firmware with the SDK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    from sundew_ipc_v1_pb2 import ScoreEvent as ProtoScoreEvent
except ModuleNotFoundError:  # pragma: no cover - optional binding
    ProtoScoreEvent = None

from ..metrics import MetricsSnapshot


@dataclass
class SundewGateShim:
    """Placeholder shim mirroring the C header functions."""

    initialized: bool = False
    last_decision: Optional[bool] = None
    last_metrics: Optional[MetricsSnapshot] = None

    def gate_init(self, request: dict) -> dict:
        self.initialized = True
        return {'status': 'OK', 'heartbeat_interval_ms': 2000}

    def gate_score(self, features: dict) -> dict:
        if not self.initialized:
            return {'status': 'ERROR', 'detail': 'not initialized'}
        self.last_decision = bool(features.get('should_activate', True))
        return {'status': 'OK', 'should_activate': self.last_decision}

    def score_event(self, event: 'ProtoScoreEvent | object') -> dict:
        """Process a generated ScoreEvent into the response dict."""
        try:
            features = {kv.key: kv.value for kv in event.features}  # type: ignore[union-attr]
        except AttributeError:
            raise TypeError('Expected ScoreEvent-like object with features')
        result = self.gate_score(features)
        result['sequence'] = getattr(event, 'sequence', 0)
        return result

    def gate_telemetry(self, snapshot: MetricsSnapshot) -> dict:
        self.last_metrics = snapshot
        return {'status': 'OK'}


__all__ = ['SundewGateShim']
