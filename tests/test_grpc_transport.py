import grpc

from sundew_core_sdk.config import SDKConfig
from sundew_core_sdk.controller import AdaptiveGateController
from sundew_core_sdk.ipc.adapter import IPCAdapter
from sundew_core_sdk.ipc.grpc_transport import SundewGateStub, serve
from sundew_core_sdk.metrics import MetricsTracker
from sundew_ipc_v1_pb2 import FeatureKV, ScoreEvent, TelemetryPush


def _adapter():
    adapter = IPCAdapter(
        controller=AdaptiveGateController(SDKConfig()),
        tracker=MetricsTracker(),
    )
    adapter.controller.load_native()
    return adapter


def test_grpc_connect_roundtrip():
    adapter = _adapter()
    server, port = serve(adapter, host='127.0.0.1', port=0)
    channel = grpc.insecure_channel(f'127.0.0.1:{port}')
    stub = SundewGateStub(channel)
    try:
        responses = stub.Connect(
            iter(
                [
                    ScoreEvent(
                        sequence=7,
                        features=[
                            FeatureKV(key='glucose_mgdl', value=142.0),
                            FeatureKV(key='roc_mgdl_min', value=-1.0),
                        ],
                    )
                ]
            )
        )
        decision = next(responses)
        assert decision.sequence == 7
    finally:
        channel.close()
        server.stop(0).wait()


def test_grpc_push_telemetry_ack():
    adapter = _adapter()
    server, port = serve(adapter, host='127.0.0.1', port=0)
    channel = grpc.insecure_channel(f'127.0.0.1:{port}')
    stub = SundewGateStub(channel)
    try:
        ack = stub.PushTelemetry(
            TelemetryPush(
                activation_rate=0.5,
                average_power_w=3.0,
                energy_buffer=90.0,
                temperature_c=32.0,
                samples=12,
            )
        )
        assert ack.sequence == 12
    finally:
        channel.close()
        server.stop(0).wait()
