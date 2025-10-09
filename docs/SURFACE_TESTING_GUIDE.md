# Surface Testing Guide - SDK Validation Without Hardware

This guide shows you how to test the Sundew SDK on your Surface laptop(s) before investing in embedded hardware.

## Prerequisites

- Surface laptop (Windows 10/11)
- Python 3.10+
- Git installed
- Internet connection

## Option 1: Single Surface Testing (Quickest)

### Step 1: Clone and Setup

```bash
# Open PowerShell or Windows Terminal
cd C:\Users\adminidiakhoa\

# Clone repository (if not already cloned)
git clone https://github.com/oluwafemidiakhoa/sundew_algorithms.git
cd sundew_algorithms

# Pull latest changes
git pull origin main

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -e .
pip install grpcio grpcio-tools protobuf psutil
```

### Step 2: Generate IPC Bindings

```bash
# Generate protobuf bindings
python tools\generate_ipc_bindings.py

# Should output:
# C:\Users\...\python.exe -m grpc_tools.protoc --proto_path=... (success)
```

### Step 3: Run SDK Demo

```bash
# Test IPC integration
python examples\ipc_demo.py

# Expected output:
# Gate decision: True/False
# Telemetry: TelemetryEvent(activation_rate=..., threshold=..., energy_level=...)
```

### Step 4: Run Test Suite

```bash
# Run all SDK tests
pytest tests\test_ipc*.py tests\test_grpc*.py -v

# Expected: 12 tests passed
```

### Step 5: Start IPC Server

```bash
# Terminal 1: Start server
python -c "
from sundew_core_sdk.config import SDKConfig
from sundew_core_sdk.controller import AdaptiveGateController
from sundew_core_sdk.ipc.adapter import IPCAdapter
from sundew_core_sdk.ipc.transport import IPCServer, IPCServerConfig
from sundew_core_sdk.metrics import MetricsTracker

adapter = IPCAdapter(
    controller=AdaptiveGateController(SDKConfig(target_activation=0.15)),
    tracker=MetricsTracker()
)
adapter.controller.load_native()
server = IPCServer(adapter, IPCServerConfig(port=8765))
server.start()
print('IPC server running on localhost:8765')
import time
while True: time.sleep(1)
"
```

### Step 6: Send Test Events

```bash
# Terminal 2: Send test events
python tools\send_score_event.py --port 8765 --feature glucose_mgdl=140
python tools\send_score_event.py --port 8765 --feature glucose_mgdl=200
python tools\send_score_event.py --port 8765 --feature glucose_mgdl=80
```

---

## Option 2: Two-Surface Testing (Simulates Real Hardware)

This simulates a real edge device + monitoring station setup.

### Surface 1 Setup (Acts as "Edge Device")

**On Surface 1:**

```bash
# 1. Clone and setup (same as above)
cd C:\Users\adminidiakhoa\sundew_algorithms
git pull
.venv\Scripts\activate
pip install grpcio grpcio-tools protobuf psutil

# 2. Find your IP address
ipconfig
# Look for "IPv4 Address" under your active network adapter
# Example: 192.168.1.100 (note this down)

# 3. Start IPC server on network (not just localhost)
python -c "
from sundew_core_sdk.config import SDKConfig
from sundew_core_sdk.controller import AdaptiveGateController
from sundew_core_sdk.ipc.adapter import IPCAdapter
from sundew_core_sdk.ipc.transport import IPCServer, IPCServerConfig
from sundew_core_sdk.metrics import MetricsTracker

adapter = IPCAdapter(
    controller=AdaptiveGateController(SDKConfig(target_activation=0.15)),
    tracker=MetricsTracker()
)
adapter.controller.load_native()
server = IPCServer(adapter, IPCServerConfig(host='0.0.0.0', port=8765))
server.start()
print('Edge device simulator running on 0.0.0.0:8765')
import time
while True: time.sleep(1)
"
```

### Surface 2 Setup (Monitoring Station)

**On Surface 2 (your main laptop):**

```bash
# 1. Clone repo (if not already present)
cd C:\Users\adminidiakhoa\sundew_algorithms
git pull
.venv\Scripts\activate

# 2. Send events to Surface 1 (replace with actual IP)
python tools\send_score_event.py --host 192.168.1.100 --port 8765 --feature glucose_mgdl=140

# 3. Run workload simulation
python benchmarks\run_dataset_suite.py --presets custom_breast_probe --out results\surface_test.csv
```

### Power Measurement (Two-Surface Method)

**On Surface 1 (unplug from AC power):**

```bash
# 1. Note battery percentage: Open Settings → System → Battery (e.g., 100%)

# 2. Run baseline workload (NO gating)
python benchmarks\run_dataset_suite.py --presets aggressive --out results\baseline.csv

# Wait 10 minutes, note battery % (e.g., 92%) = 8% drain in 10 min

# 3. Recharge to 100%, then run gated workload
python benchmarks\run_dataset_suite.py --presets conservative --out results\gated.csv

# Wait 10 minutes, note battery % (e.g., 96%) = 4% drain in 10 min
# Energy savings: (8% - 4%) / 8% = 50% savings!
```

**Using Task Manager Power Monitoring:**

1. Open Task Manager (Ctrl+Shift+Esc)
2. Go to "Performance" tab → CPU
3. Watch power consumption while running workloads
4. Compare aggressive vs conservative presets

---

## Option 3: Windows Power Monitoring Script

```bash
# Run power monitoring during SDK testing
python -c "
import psutil
import time

battery = psutil.sensors_battery()
if battery:
    print(f'Battery: {battery.percent:.1f}%')
    print(f'Plugged in: {battery.power_plugged}')
    print(f'Time remaining: {battery.secsleft // 60} minutes')

    if battery.power_plugged:
        print('\nWARNING: Unplug Surface from AC for accurate power testing!')
    else:
        print('\nReady for power measurement testing.')
else:
    print('No battery detected - using desktop?')
"
```

---

## Validation Checklist

After completing the tests, verify:

- [ ] IPC bindings generated successfully
- [ ] `examples\ipc_demo.py` runs without errors
- [ ] All 12 SDK tests pass (`pytest tests\test_ipc*.py tests\test_grpc*.py`)
- [ ] IPC server accepts connections on port 8765
- [ ] `send_score_event.py` successfully sends events
- [ ] Two-Surface network communication works (if testing)
- [ ] Battery drain is measurably lower with conservative preset (if testing)

## Troubleshooting

### "ModuleNotFoundError: No module named 'grpc_tools'"

```bash
pip install grpcio-tools
```

### "Port 8765 already in use"

```bash
# Find and kill process using port
netstat -ano | findstr :8765
taskkill /PID <PID_NUMBER> /F
```

### "Connection refused" (two-Surface testing)

1. Check Windows Firewall - allow port 8765
2. Verify both Surfaces on same WiFi network
3. Ping Surface 1 from Surface 2: `ping 192.168.1.100`

### IPC Demo shows "Gate decision: False" always

This is normal - the controller makes probabilistic decisions. Run multiple times:

```bash
for /L %i in (1,1,10) do python examples\ipc_demo.py
```

## Next Steps

After Surface validation:
1. ✅ SDK works correctly
2. Try Google Colab testing (see `docs\COLAB_TESTING_GUIDE.md`)
3. Deploy to real hardware (Raspberry Pi, Jetson) when ready
4. Integrate INA219 power sensor for accurate measurements

## Performance Expectations

On Surface laptop:
- **Baseline (aggressive preset)**: ~10-15W CPU power, high battery drain
- **Gated (conservative preset)**: ~5-8W CPU power, 40-60% less battery drain
- **Activation rate**: 10-20% with conservative, 80-90% with aggressive

This validates the SDK is working correctly before hardware deployment!
