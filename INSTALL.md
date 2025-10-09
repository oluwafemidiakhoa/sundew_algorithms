# Sundew Algorithms - Installation Guide

This guide covers all installation scenarios for the Sundew SDK.

## Table of Contents
- [Quick Start](#quick-start)
- [Installation Options](#installation-options)
- [Platform-Specific Setup](#platform-specific-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Basic Installation (Core Algorithm Only)

```bash
# Clone repository
git clone https://github.com/oluwafemidiakhoa/sundew_algorithms.git
cd sundew_algorithms

# Install core package
pip install -e .
```

This installs only the core Sundew algorithm without SDK/hardware dependencies.

### SDK Installation (For Hardware Deployment)

```bash
# Install with SDK dependencies
pip install -e ".[sdk]"

# Or using requirements file
pip install -r requirements-sdk.txt
```

This includes gRPC, protobuf, and IPC layer for embedded device deployment.

---

## Installation Options

### Option 1: Using uv (Recommended for Development)

```bash
# Install uv package manager
pip install uv

# Install core package
uv pip install -e .

# Install SDK dependencies
uv pip install grpcio grpcio-tools protobuf

# Or install development dependencies
uv pip install -e ".[dev]"
```

### Option 2: Using pip with Optional Dependencies

```bash
# Core only
pip install -e .

# SDK for hardware deployment
pip install -e ".[sdk]"

# Development (includes testing, linting)
pip install -e ".[dev]"

# Hardware (includes sensors, GPIO)
pip install -e ".[hardware]"

# All dependencies
pip install -e ".[all]"
```

### Option 3: Using Requirements Files

```bash
# For Surface/laptop testing
pip install -r requirements-sdk.txt

# For development
pip install -r requirements-dev.txt

# For hardware deployment (Raspberry Pi, Jetson)
pip install -r requirements-hardware.txt

# For Google Colab
pip install -r requirements-colab.txt
```

---

## Platform-Specific Setup

### Windows (Surface Laptop)

```powershell
# Open PowerShell or Windows Terminal
cd C:\Users\<username>\sundew_algorithms

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install SDK
pip install -e ".[sdk]"

# Generate IPC bindings
python tools\generate_ipc_bindings.py

# Test installation
python examples\ipc_demo.py
```

**See:** [docs/SURFACE_TESTING_GUIDE.md](docs/SURFACE_TESTING_GUIDE.md) for detailed Windows setup.

### Google Colab

```python
# In a Colab notebook cell
!git clone https://github.com/oluwafemidiakhoa/sundew_algorithms.git
%cd sundew_algorithms
!pip install -q -e .
!pip install -q grpcio grpcio-tools protobuf
!python tools/generate_ipc_bindings.py
```

**See:** [docs/COLAB_TESTING_GUIDE.md](docs/COLAB_TESTING_GUIDE.md) for full Colab workflow.

### Raspberry Pi 4 / Compute Module

```bash
# SSH into Raspberry Pi
ssh pi@raspberrypi.local

# Update system
sudo apt update
sudo apt install -y python3-pip python3-venv git i2c-tools

# Enable I2C
sudo raspi-config
# Interface Options → I2C → Enable

# Clone and install
git clone https://github.com/oluwafemidiakhoa/sundew_algorithms.git
cd sundew_algorithms
python3 -m venv .venv
source .venv/bin/activate

# Install hardware dependencies
pip install -e ".[hardware]"

# Generate bindings
python tools/generate_ipc_bindings.py

# Verify I2C (if using INA219 sensor)
sudo i2cdetect -y 1
```

**See:** [docs/sdk/ipc_quickstart.md](docs/sdk/ipc_quickstart.md) for hardware deployment.

### NVIDIA Jetson Nano / Orin

```bash
# SSH into Jetson
ssh nvidia@jetson.local

# Install dependencies
sudo apt update
sudo apt install -y python3-pip python3-venv git

# Clone and install
git clone https://github.com/oluwafemidiakhoa/sundew_algorithms.git
cd sundew_algorithms
python3 -m venv .venv
source .venv/bin/activate

# Install SDK
pip install -e ".[sdk]"

# For power monitoring (if using INA3221)
pip install adafruit-circuitpython-ina3221

# Generate bindings
python tools/generate_ipc_bindings.py
```

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt update
sudo apt install -y python3-pip python3-venv git

# Clone and install
git clone https://github.com/oluwafemidiakhoa/sundew_algorithms.git
cd sundew_algorithms
python3 -m venv .venv
source .venv/bin/activate

# Install SDK
pip install -e ".[sdk]"

# Generate bindings
python tools/generate_ipc_bindings.py
```

---

## Verification

### Verify Core Installation

```bash
# Test Python import
python -c "from sundew import SundewAlgorithm; print('Core installed ✓')"

# Run basic algorithm test
pytest tests/test_core.py -v
```

### Verify SDK Installation

```bash
# Generate IPC bindings
python tools/generate_ipc_bindings.py

# Run IPC demo
python examples/ipc_demo.py
# Expected output: "Gate decision: True/False"

# Run SDK test suite
pytest tests/test_ipc*.py tests/test_grpc*.py -v
# Expected: 12 tests passed
```

### Verify Hardware Installation (Raspberry Pi)

```bash
# Check I2C sensor
sudo i2cdetect -y 1
# Should show device at 0x40 (INA219)

# Test power measurement
python benchmarks/power/capture_power.py --duration 10 --output test.csv
# Should create test.csv with power readings
```

---

## Dependencies Summary

### Core Dependencies (Always Installed)
- `numpy>=1.22` - Numerical computations
- `pandas>=1.5.0` - Data manipulation

### SDK Dependencies (Optional: `[sdk]`)
- `grpcio>=1.75.0` - gRPC framework
- `grpcio-tools>=1.75.0` - Protobuf compiler
- `protobuf>=6.32.0` - Protocol buffers
- `psutil>=5.9.0` - System monitoring

### Hardware Dependencies (Optional: `[hardware]`)
- `adafruit-circuitpython-ina219` - INA219 power sensor
- `adafruit-blinka` - CircuitPython hardware API
- `smbus2` - I2C communication
- `RPi.GPIO` - Raspberry Pi GPIO (Linux only)

### Development Dependencies (Optional: `[dev]`)
- `pytest>=7` - Testing framework
- `ruff>=0.13` - Linting and formatting
- `mypy>=1.7` - Type checking
- `hypothesis>=6` - Property-based testing

---

## Troubleshooting

### "No module named 'grpc_tools'"

```bash
pip install grpcio-tools
```

### "Permission denied" (Linux/Raspberry Pi)

```bash
# Use virtual environment instead of --user
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[sdk]"
```

### "I2C device not found" (Raspberry Pi)

```bash
# Enable I2C
sudo raspi-config
# Interface Options → I2C → Enable

# Reboot
sudo reboot

# Check connection
sudo i2cdetect -y 1
```

### "protobuf version mismatch"

```bash
# Reinstall matching versions
pip install --upgrade grpcio==1.75.1 grpcio-tools==1.75.1 protobuf==6.32.1
python tools/generate_ipc_bindings.py
```

### Windows: "python: command not found"

```powershell
# Use py launcher
py -m pip install -e ".[sdk]"
py tools/generate_ipc_bindings.py
```

---

## Next Steps

After installation:

1. **Surface Testing**: [docs/SURFACE_TESTING_GUIDE.md](docs/SURFACE_TESTING_GUIDE.md)
2. **Colab Testing**: [docs/COLAB_TESTING_GUIDE.md](docs/COLAB_TESTING_GUIDE.md)
3. **Hardware Deployment**: [docs/sdk/ipc_quickstart.md](docs/sdk/ipc_quickstart.md)
4. **SDK Documentation**: [docs/sdk/README.md](docs/sdk/README.md)

---

## Support

- **Issues**: https://github.com/oluwafemidiakhoa/sundew_algorithms/issues
- **Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)
