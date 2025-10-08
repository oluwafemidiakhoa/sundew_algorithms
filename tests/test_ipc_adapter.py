
import pytest

from sundew_core_sdk.config import SDKConfig
from sundew_core_sdk.controller import AdaptiveGateController
from sundew_core_sdk.ipc.adapter import IPCAdapter
from sundew_core_sdk.metrics import MetricsTracker

try:
    from sundew_ipc_v1_pb2 import FeatureKV, ScoreEvent, TelemetryPush
except Exception as exc:  # pragma: no cover - missing bindings
    FeatureKV = ScoreEvent = TelemetryPush = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


def _require_bindings():
    if IMPORT_ERROR:
        pytest.skip(f"IPC protobuf bindings unavailable: {IMPORT_ERROR}")


def _adapter():
    adapter = IPCAdapter(
        controller=AdaptiveGateController(SDKConfig()),
        tracker=MetricsTracker(),
    )
    adapter.controller.load_native()
    return adapter


def test_handle_score_event_sets_sequence():
    _require_bindings()
    adapter = _adapter()
    event = ScoreEvent(sequence=42, features=[FeatureKV(key='glucose_mgdl', value=140.0)])
    decision = adapter.handle_score_event(event)
    assert decision.sequence == 42


def test_record_telemetry_returns_event():
    _require_bindings()
    adapter = _adapter()
    push = TelemetryPush(activation_rate=0.4, average_power_w=3.2)
    telemetry = adapter.record_telemetry(push)
    assert telemetry.activation_rate >= 0.0
    assert telemetry.energy_level == adapter.tracker.energy_buffer
