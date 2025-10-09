# Google Colab Testing Guide - Free Cloud SDK Validation

Test the Sundew SDK in Google Colab without any hardware. This is perfect for validating the SDK before deploying to real devices.

## Quick Start

### Method 1: Direct Notebook Link (Easiest)

1. **Open the Colab Notebook:**
   - Go to: https://colab.research.google.com/
   - Click "File" → "Open notebook"
   - Go to "GitHub" tab
   - Enter: `oluwafemidiakhoa/sundew_algorithms`
   - Select: `notebooks/Sundew_SDK_Demo.ipynb` (we'll create this)

### Method 2: Manual Setup in Colab

**Step 1: Open a new Colab notebook**
- Go to https://colab.research.google.com/
- Click "New notebook"

**Step 2: Run this in the first cell:**

```python
# Cell 1: Setup environment
!git clone https://github.com/oluwafemidiakhoa/sundew_algorithms.git
%cd sundew_algorithms
!pip install -q -e .
!pip install -q grpcio grpcio-tools protobuf
```

**Step 3: Generate IPC bindings:**

```python
# Cell 2: Generate protobuf bindings
!python tools/generate_ipc_bindings.py
print("✅ IPC bindings generated")
```

**Step 4: Run SDK demo:**

```python
# Cell 3: Run IPC demo
!python examples/ipc_demo.py
```

**Step 5: Run test suite:**

```python
# Cell 4: Run SDK tests
!pip install -q pytest pytest-cov hypothesis
!pytest tests/test_ipc*.py tests/test_grpc*.py -v
```

**Step 6: Simulate power workload:**

```python
# Cell 5: Run simulated power benchmark
!python benchmarks/power/run_simulated_workload.py --duration 60 --preset custom_breast_probe
```

---

## Complete Colab Notebook

Copy this entire notebook into Colab:

### Cell 1: Environment Setup

```python
"""
Sundew SDK Validation in Google Colab
This notebook validates the Sundew Core SDK without requiring physical hardware.
"""

import sys
import os

# Clone repository
if not os.path.exists("sundew_algorithms"):
    !git clone https://github.com/oluwafemidiakhoa/sundew_algorithms.git
    %cd sundew_algorithms
else:
    %cd sundew_algorithms
    !git pull origin main

# Install dependencies
!pip install -q numpy pandas grpcio grpcio-tools protobuf

print("✅ Environment setup complete")
```

### Cell 2: Generate IPC Bindings

```python
# Generate protobuf bindings
!python tools/generate_ipc_bindings.py

# Verify bindings were created
import os
if os.path.exists("src/sundew_ipc_v1_pb2.py"):
    print("✅ IPC bindings generated successfully")
else:
    print("❌ Binding generation failed")
```

### Cell 3: Run IPC Demo

```python
# Run the IPC demo
!python examples/ipc_demo.py
```

### Cell 4: SDK Test Suite

```python
# Install testing dependencies
!pip install -q pytest pytest-cov hypothesis

# Run all SDK tests
!pytest tests/test_ipc*.py tests/test_grpc*.py -v --tb=short
```

### Cell 5: Interactive SDK Testing

```python
# Import SDK components
from sundew_core_sdk import SDKConfig, AdaptiveGateController, MetricsTracker
from sundew_core_sdk.ipc.adapter import IPCAdapter
from sundew_ipc_v1_pb2 import ScoreEvent, FeatureKV

# Initialize SDK
config = SDKConfig(target_activation=0.15, gate_temperature=0.08)
controller = AdaptiveGateController(config)
controller.load_native()

adapter = IPCAdapter(controller=controller, tracker=MetricsTracker())

# Create test event
event = ScoreEvent(
    sequence=1,
    features=[
        FeatureKV(key="glucose_mgdl", value=150.0),
        FeatureKV(key="heart_rate", value=75.0),
    ]
)

# Process event
decision = adapter.handle_score_event(event)
print(f"Gate Decision: {decision.should_activate}")
print(f"Confidence: {decision.confidence}")

# Get metrics
snapshot = adapter.tracker.snapshot()
print(f"Activation Rate: {snapshot.activation_rate:.2%}")
print(f"Samples Processed: {snapshot.samples}")
```

### Cell 6: Benchmark Different Presets

```python
# Compare different presets
from sundew.config_presets import get_preset

presets = ["tuned_v2", "aggressive", "conservative", "custom_breast_probe"]

print("Preset Comparison:")
print("-" * 60)

for preset_name in presets:
    preset = get_preset(preset_name)
    print(f"\n{preset_name}:")
    print(f"  Activation Threshold: {preset.activation_threshold}")
    print(f"  Target Activation Rate: {preset.target_activation_rate}")
    print(f"  Energy Pressure: {preset.energy_pressure}")
    print(f"  Gate Temperature: {preset.gate_temperature}")
```

### Cell 7: Simulated Power Workload

```python
# Run simulated workload with different presets
import subprocess
import json

results = {}

for preset in ["aggressive", "conservative"]:
    print(f"\n{'='*60}")
    print(f"Running {preset} preset...")
    print('='*60)

    # Run simulated workload
    subprocess.run([
        sys.executable,
        "benchmarks/power/run_simulated_workload.py",
        "--duration", "30",
        "--preset", preset,
        "--output", f"results_{preset}.json"
    ])

    # Load results
    try:
        with open(f"results_{preset}.json") as f:
            results[preset] = json.load(f)
    except FileNotFoundError:
        print(f"Results file not found for {preset}")

# Compare results
if results:
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)

    for preset, data in results.items():
        print(f"\n{preset.upper()}:")
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"  {key}: {value}")
```

### Cell 8: IPC Server Simulation

```python
# Start IPC server in background (simulated)
from threading import Thread
from sundew_core_sdk.ipc.transport import IPCServer, IPCServerConfig

# Create server
adapter = IPCAdapter(
    controller=AdaptiveGateController(SDKConfig()),
    tracker=MetricsTracker()
)
adapter.controller.load_native()

server = IPCServer(adapter, IPCServerConfig(port=8765))

# Start server in background
def run_server():
    server.start()
    import time
    time.sleep(30)  # Run for 30 seconds
    server.stop()

thread = Thread(target=run_server, daemon=True)
thread.start()

print("✅ IPC server started on port 8765 (running for 30 seconds)")
print("You can now send events to localhost:8765")

# Wait a moment for server to start
import time
time.sleep(2)
```

### Cell 9: Send Events to IPC Server

```python
# Send test events to the running server
import socket
import json

def send_event(host="127.0.0.1", port=8765, features=None):
    """Send a ScoreEvent to the IPC server."""
    if features is None:
        features = {"glucose_mgdl": 140.0}

    payload = json.dumps({
        "type": "score_event",
        "event": {
            "sequence": 1,
            "features": [{"key": k, "value": v} for k, v in features.items()]
        }
    }).encode()

    try:
        with socket.create_connection((host, port), timeout=5) as sock:
            sock.sendall(payload)
            response = sock.recv(4096).decode()
            result = json.loads(response)
            print(f"Features: {features}")
            print(f"Decision: {'ACTIVATE' if result.get('should_activate') else 'SKIP'}")
            print(f"Sequence: {result.get('sequence')}")
            print("-" * 40)
            return result
    except Exception as e:
        print(f"Error: {e}")
        return None

# Send multiple test events
test_cases = [
    {"glucose_mgdl": 140.0},
    {"glucose_mgdl": 200.0},
    {"glucose_mgdl": 80.0},
    {"heart_rate": 120.0},
    {"temperature": 38.5},
]

print("Sending test events to IPC server:")
print("=" * 40)

for features in test_cases:
    send_event(features=features)
    time.sleep(0.5)
```

### Cell 10: Visualization (Optional)

```python
# Visualize activation decisions
!pip install -q matplotlib

import matplotlib.pyplot as plt
import numpy as np

# Simulate activation pattern
np.random.seed(42)
events = 100
activations = []

for i in range(events):
    event = ScoreEvent(
        sequence=i,
        features=[FeatureKV(key="signal", value=np.random.randn())]
    )
    decision = adapter.handle_score_event(event)
    activations.append(1 if decision.should_activate else 0)

# Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(activations, 'o-', markersize=3)
plt.title('Activation Pattern')
plt.xlabel('Event Number')
plt.ylabel('Activated (1=Yes, 0=No)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
cumulative_activation = np.cumsum(activations) / np.arange(1, events + 1)
plt.plot(cumulative_activation)
plt.title('Cumulative Activation Rate')
plt.xlabel('Event Number')
plt.ylabel('Activation Rate')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nFinal Activation Rate: {np.mean(activations):.2%}")
print(f"Energy Savings Estimate: {(1 - np.mean(activations)):.2%}")
```

---

## Expected Output

### IPC Demo Cell:
```
Gate decision: False
Telemetry: TelemetryEvent(activation_rate=0.5, threshold=0.0, energy_level=0.0)
```

### Test Suite Cell:
```
============================= test session starts ==============================
tests/test_ipc_adapter.py::test_handle_score_event_sets_sequence PASSED  [ 8%]
tests/test_ipc_adapter.py::test_record_telemetry_returns_event PASSED    [16%]
...
============================= 12 passed in 0.82s ===============================
```

### Activation Pattern:
You should see a visualization showing sporadic activations with overall ~15-20% activation rate (for conservative preset).

---

## Troubleshooting in Colab

### "No module named 'sundew'"
```python
# Reinstall package
!pip install -e . --force-reinstall
```

### "protoc command not found"
```python
# Reinstall grpcio-tools
!pip install --upgrade grpcio-tools
```

### Server connection timeout
```python
# Increase timeout or check if server thread started
import time
time.sleep(5)  # Give server more time to start
```

---

## What This Validates

✅ **SDK Installation**: All dependencies install correctly
✅ **IPC Bindings**: Protobuf generation works
✅ **Core Functionality**: Controller makes gating decisions
✅ **Transport Layer**: TCP/socket communication functional
✅ **Metrics Tracking**: Telemetry collection works
✅ **Test Coverage**: All 12 SDK tests pass

## Next Steps After Colab

1. ✅ SDK validated in cloud environment
2. Test on Surface laptop (see `docs/SURFACE_TESTING_GUIDE.md`)
3. Deploy to real hardware when ready
4. Integrate power sensors for real measurements

## Saving Your Colab Notebook

1. File → Save a copy in Drive
2. File → Download → Download .ipynb
3. Or: Connect to GitHub and save directly to your repo

## Running on Colab TPU (Advanced)

```python
# Check if TPU is available
import os
if 'COLAB_TPU_ADDR' in os.environ:
    print("✅ TPU available")
    # Can simulate Coral Edge TPU workloads here
else:
    print("ℹ️ No TPU - using CPU (this is fine for SDK testing)")
```

---

**Ready to test!** Open https://colab.research.google.com/ and paste the cells above.
