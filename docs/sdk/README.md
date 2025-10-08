# Sundew Core SDK Documentation

The Sundew Core SDK enables hardware deployment of bio-inspired energy-aware gating on edge devices (Jetson Nano, Coral Edge TPU, Raspberry Pi, etc.).

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/oluwafemidiakhoa/sundew_algorithms.git
cd sundew_algorithms

# Install with gRPC support
pip install -e .
pip install grpcio grpcio-tools protobuf

# Generate IPC bindings
python tools/generate_ipc_bindings.py
```

### Basic Usage

```python
from sundew_core_sdk import SDKConfig, AdaptiveGateController, MetricsTracker
from sundew_core_sdk.ipc.adapter import IPCAdapter

# Initialize SDK
config = SDKConfig(target_activation=0.15, gate_temperature=0.08)
controller = AdaptiveGateController(config)
controller.load_native()

# Create IPC adapter
adapter = IPCAdapter(controller=controller, tracker=MetricsTracker())

# Process events
from sundew_ipc_v1_pb2 import ScoreEvent, FeatureKV
event = ScoreEvent(sequence=1, features=[
    FeatureKV(key="glucose_mgdl", value=150.0)
])
decision = adapter.handle_score_event(event)
print(f"Activate: {decision.should_activate}")
```

### Running IPC Server

```bash
# Start TCP server on port 8765
python -c "
from sundew_core_sdk.config import SDKConfig
from sundew_core_sdk.controller import AdaptiveGateController
from sundew_core_sdk.ipc.adapter import IPCAdapter
from sundew_core_sdk.ipc.transport import IPCServer, IPCServerConfig
from sundew_core_sdk.metrics import MetricsTracker

adapter = IPCAdapter(
    controller=AdaptiveGateController(SDKConfig()),
    tracker=MetricsTracker()
)
adapter.controller.load_native()
server = IPCServer(adapter, IPCServerConfig(port=8765))
server.start()
print('Server running on port 8765')
import time
while True: time.sleep(1)
"
```

### Testing with Demo Script

```bash
# Run IPC integration demo
python examples/ipc_demo.py

# Send test events to server
python tools/send_score_event.py --port 8765 --feature glucose_mgdl=140
```

## Architecture

### Core Components

- **[SDKConfig](../../src/sundew_core_sdk/config.py)** - Configuration management
- **[AdaptiveGateController](../../src/sundew_core_sdk/controller.py)** - Gating decision engine
- **[MetricsTracker](../../src/sundew_core_sdk/metrics.py)** - Performance metrics collection
- **[TelemetryEvent](../../src/sundew_core_sdk/telemetry.py)** - Telemetry data structures

### IPC Layer

- **[IPCAdapter](../../src/sundew_core_sdk/ipc/adapter.py)** - Protobuf ↔ SDK bridge
- **[IPCServer](../../src/sundew_core_sdk/ipc/transport.py)** - TCP/Unix socket transport
- **[gRPC Transport](../../src/sundew_core_sdk/ipc/grpc_transport.py)** - Production gRPC service
- **[Bindings](../../src/sundew_core_sdk/ipc/bindings.py)** - Protobuf loaders

### Data Flow

```
Firmware/Client → ScoreEvent (protobuf)
                ↓
            IPCAdapter
                ↓
        AdaptiveGateController
                ↓
            GateDecision (protobuf) → Back to client
```

## Hardware Integration

### Supported Platforms

- Raspberry Pi 4B / Compute Module 4
- NVIDIA Jetson Nano / Orin Nano
- Google Coral Edge TPU
- x86 Linux (Ubuntu, Debian)
- Windows (development/testing)

### Power Measurement

See [power_capture.md](power_capture.md) for INA219/INA3221 sensor integration.

```bash
# Capture power trace
python benchmarks/power/capture_power.py \
  --duration 300 \
  --interval 0.1 \
  --output power_baseline.csv
```

### Deployment

See [deployment_plan.md](deployment_plan.md) and [ipc_quickstart.md](ipc_quickstart.md) for on-device setup.

## API Reference

### SDKConfig

```python
@dataclass
class SDKConfig:
    target_activation: float = 0.22      # Target activation rate (0-1)
    gate_temperature: float = 0.08       # Gating exploration temperature
    max_energy: float = 100.0            # Energy budget
    firmware_endpoint: Optional[str] = None  # Firmware connection string
```

### AdaptiveGateController

```python
controller = AdaptiveGateController(config)
controller.load_native()                 # Load Sundew algorithm
decision = controller.decide(features)   # Returns bool (activate?)
telemetry = controller.emit_telemetry()  # Get current metrics
```

### MetricsTracker

```python
tracker = MetricsTracker(window=2048)
tracker.record(activated=True, power_w=2.4)
snapshot = tracker.snapshot()
print(snapshot.activation_rate, snapshot.avg_power_w)
```

## Testing

```bash
# Run all SDK tests
pytest tests/test_ipc*.py tests/test_grpc*.py -v

# Test with coverage
pytest tests/test_ipc*.py --cov=src/sundew_core_sdk --cov-report=html
```

## Troubleshooting

### Import Errors

```bash
# Ensure bindings are generated
python tools/generate_ipc_bindings.py

# Check for version mismatch
pip list | grep grpcio
pip list | grep protobuf
```

### gRPC Connection Issues

```bash
# Check server is running
netstat -an | grep 8765  # Unix
netstat -an | findstr 8765  # Windows

# Test with client
python tools/send_score_event.py --host 127.0.0.1 --port 8765
```

### Performance Issues

- Lower `target_activation` for more energy savings
- Increase `gate_temperature` for more exploration
- Check metrics with `MetricsTracker.snapshot()`

## Examples

- [examples/ipc_demo.py](../../examples/ipc_demo.py) - Basic IPC integration
- [benchmarks/power/](../../benchmarks/power/) - Power measurement workflows
- [tools/send_score_event.py](../../tools/send_score_event.py) - CLI client

## Documentation Files

- `architecture.md` — high-level design and module interactions
- `hardware.md` — board-specific bring-up guidance
- `benchmarks.md` — methodology for measuring power savings
- `firmware.md` — integration details for the embedded shim
- `ipc_quickstart.md` — on-device deployment quickstart
- `power_capture.md` — power measurement setup

## Phase 1 Status

✅ **Completed:**
- SDK core components (config, controller, metrics, telemetry)
- IPC layer (adapter, shim, transport, gRPC)
- Protobuf bindings generation
- Comprehensive test suite (12 tests passing)
- Demo applications

⏳ **In Progress:**
- Hardware adapter implementations
- On-device deployment automation
- Real power measurement validation

## Support

- Issues: https://github.com/oluwafemidiakhoa/sundew_algorithms/issues
- Phase 1 Plan: [../phase1_sundew_core_sdk_plan.md](../phase1_sundew_core_sdk_plan.md)
