
from sundew_core_sdk.ipc.shim import SundewGateShim
from sundew_core_sdk.metrics import MetricsSnapshot


def test_gate_init_sets_state():
    shim = SundewGateShim()
    resp = shim.gate_init({"board": "test"})
    assert shim.initialized is True
    assert resp["status"] == "OK"


def test_gate_score_requires_init():
    shim = SundewGateShim()
    resp = shim.gate_score({"should_activate": True})
    assert resp["status"] == "ERROR"
    shim.gate_init({})
    resp = shim.gate_score({"should_activate": False})
    assert resp["status"] == "OK"
    assert shim.last_decision is False


def test_gate_telemetry_records_snapshot():
    shim = SundewGateShim()
    shim.gate_init({})
    snapshot = MetricsSnapshot(
        activation_rate=0.5,
        avg_power_w=2.0,
        energy_buffer=80.0,
        samples=10,
    )
    ack = shim.gate_telemetry(snapshot)
    assert ack["status"] == "OK"
    assert shim.last_metrics == snapshot




try:
    from sundew_ipc_v1_pb2 import FeatureKV, ScoreEvent
except Exception as exc:  # pragma: no cover - runtime mismatch
    FeatureKV = ScoreEvent = None
    SCORE_EVENT_IMPORT_ERROR = exc
else:
    SCORE_EVENT_IMPORT_ERROR = None


def test_shim_score_event_converts_features():
    if SCORE_EVENT_IMPORT_ERROR:
        import pytest

        pytest.skip(f"ScoreEvent unavailable: {SCORE_EVENT_IMPORT_ERROR}")
    shim = SundewGateShim()
    shim.gate_init({})
    event = ScoreEvent(sequence=7, features=[FeatureKV(key='foo', value=1.5)])
    resp = shim.score_event(event)
    assert resp['sequence'] == 7
    assert resp['status'] == 'OK'
    assert shim.last_decision is True
