
"""Minimal in-process IPC demo linking adapter, shim, and controller."""

from __future__ import annotations

from sundew_core_sdk.config import SDKConfig
from sundew_core_sdk.controller import AdaptiveGateController
from sundew_core_sdk.ipc.adapter import IPCAdapter
from sundew_core_sdk.ipc.shim import SundewGateShim
from sundew_core_sdk.metrics import MetricsTracker

from sundew_ipc_v1_pb2 import FeatureKV, ScoreEvent, TelemetryPush


def main() -> None:
    shim = SundewGateShim()
    shim.gate_init({"board": "demo"})

    adapter = IPCAdapter(
        controller=AdaptiveGateController(SDKConfig()),
        tracker=MetricsTracker(),
    )
    adapter.controller.load_native()

    event = ScoreEvent(
        sequence=1,
        features=[FeatureKV(key="glucose_mgdl", value=150.0)],
    )
    decision = adapter.handle_score_event(event)
    print("Gate decision:", decision.should_activate)

    telemetry = adapter.record_telemetry(
        TelemetryPush(activation_rate=0.3, average_power_w=2.4)
    )
    print("Telemetry:", telemetry)


if __name__ == "__main__":
    main()
