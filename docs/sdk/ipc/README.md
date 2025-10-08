# Sundew IPC Binding Instructions

To regenerate protobuf bindings once `grpcio-tools` is available, run:

```bash
pip install grpcio grpcio-tools
python tools/generate_ipc_bindings.py
```

The command must be run from a shell where `grpc_tools.protoc` is available. If the module is missing,
the script will print a helpful error (as seen on this machine). The repository ships with a minimal
`src/sundew_ipc_v1_pb2.py` stub so development can proceed before the official
bindings are generated. Replace these stubs with the generated files for production builds.


Stubs included for development:
- `src/sundew_ipc_v1_pb2.py`
- `src/sundew_ipc_v1_pb2_grpc.py`


## Using the bindings
With the generated files on disk you can load them via the SDK helpers:

```python
from sundew_core_sdk.ipc.bindings import load_proto_module, load_grpc_module

proto = load_proto_module()
grpc = load_grpc_module()
assert proto.ScoreEvent  # available generated message
```

If your active Python runtime ships an older `protobuf` package than the
`grpcio-tools` version used for generation, the loader will raise
`google.protobuf.runtime_version.VersionError`. Our unit tests skip in that
scenario; upgrade `protobuf` to match the gencode version if you need to run
against the actual modules.


## CI automation
Add a pipeline step to run `tools/generate_ipc_bindings.py --check` (future flag) or
validate that generated files are up to date. Tests should exercise `tests/test_ipc_bindings.py`
and `tests/test_ipc_shim.py` to ensure loaders remain healthy.

CI rule suggestion:
- run `python tools/check_ipc_bindings.py` to ensure bindings exist.
- run `pytest tests/test_ipc_bindings.py tests/test_ipc_shim.py tests/test_ipc_adapter.py`.

## Firmware integration agenda
- Finalize shared-library shim (C) with exported functions matching `sundew_ipc_v1.h`.
- Implement IPCAdapter-based daemon that proxies between firmware transport and SDK controller.
- Bring up hardware targets (Jetson Nano, Coral TPU, Raspberry Pi CM) with power instrumentation.
- Run `benchmarks/power` workloads through the IPC layer to validate 60–80% savings.


## Hardware prerequisites (Phase 1 demo)
- Jetson Nano developer kit with access to onboard power monitor or external INA219 shield.
- Coral Edge TPU USB accelerator + host SBC with USB power telemetry.
- Raspberry Pi Compute Module carrier with shunt-based power measurement.
- Stable 5V supplies, cabling, and logging laptop running the IPC daemon.
- Optional: logic analyzer or oscilloscope for validating gating signal timing.


## Transport prototype plan
1. Implement local Unix domain socket service wrapping `IPCAdapter` (fallback to TCP on Windows).
2. Integrate gRPC service generated from `sundew_ipc_v1.proto` calling into adapter.
3. Add CLI (`sundew-ipc-daemon`) taking `--transport` flag, launching shim bridge.
4. Extend tests with loopback client replaying `ScoreEvent` messages via transport.


Transport prototype (`IPCServer`) listens on Unix sockets when available, otherwise falls back to TCP (`127.0.0.1`). A basic test (`tests/test_ipc_transport.py`) exercises the JSON framing end-to-end.

Example:
```python
from sundew_core_sdk.config import SDKConfig
from sundew_core_sdk.controller import AdaptiveGateController
from sundew_core_sdk.ipc.adapter import IPCAdapter
from sundew_core_sdk.ipc.transport import IPCServer, IPCServerConfig
from sundew_core_sdk.metrics import MetricsTracker

adapter = IPCAdapter(AdaptiveGateController(SDKConfig()), MetricsTracker())
adapter.controller.load_native()
server = IPCServer(adapter, IPCServerConfig(port=8765))
server.start()
```
A client can send a JSON payload `{"type": "score_event", "event": {...}}` over TCP to receive a gate decision.
```
python - <<'PY'
import json, socket
payload = json.dumps({
    "type": "score_event",
    "event": {"sequence": 1, "features": [{"key": "glucose_mgdl", "value": 140.0}]},
}).encode()
with socket.create_connection(("127.0.0.1", 8765)) as client:
    client.sendall(payload)
    print(client.recv(4096).decode())
PY
```


CLI client: run `python tools/send_score_event.py --port 8765 --feature glucose_mgdl=140` to send events to the transport.

## gRPC transport
- Service implementation: `sundew_core_sdk.ipc.grpc_transport` exposes `serve()` returning a running grpc.Server.
- Tests: `tests/test_grpc_transport.py` exercises both `Connect` streaming and `PushTelemetry` RPCs.
- Clients can use the generated `SundewGateStub`:
```python
import grpc
from sundew_ipc_v1_pb2 import ScoreEvent, FeatureKV
from sundew_ipc_v1_pb2_grpc import SundewGateStub

channel = grpc.insecure_channel('127.0.0.1:50051')
stub = SundewGateStub(channel)
responses = stub.Connect(iter([ScoreEvent(sequence=1, features=[FeatureKV(key='glucose_mgdl', value=140.0)])]))
print(next(responses))
```
```
channel.close()
```


Firmware shim should expose the gRPC service by embedding `serve(adapter)` and bridging to native transports. Documented in examples/ipc_demo.py for initial flow.

Power log analysis: after collecting baseline/gated CSVs run `python benchmarks/power/compare_runs.py --baseline baseline.csv --gated gated.csv` to compute savings.
