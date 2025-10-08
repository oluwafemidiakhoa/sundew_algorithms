# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sundew Algorithms is a Python package implementing bio-inspired, energy-aware adaptive gating for stream processing. The core algorithm uses PI control, energy pressure, and probabilistic gating to balance processing efficiency with energy consumption in edge AI systems.

## Package Management & Dependencies

This project uses **uv** as the package manager:
- `uv run python <script>` - Run Python scripts with project dependencies
- `uv run pytest` - Run tests
- `uv pip install <package>` - Install additional packages
- Package configuration in `pyproject.toml`
- Lock file: `uv.lock`

Core dependencies: numpy, pandas
Dev dependencies: ruff, mypy, pytest, pytest-cov, hypothesis, build, twine
SDK dependencies: grpcio, grpcio-tools, protobuf (for hardware IPC)

## Competition Context

This codebase contains Sundew Algorithms, a bio-inspired energy-aware gating system. While primarily designed for edge AI and stream processing, the algorithms can be adapted for various ML competitions including NLP tasks like the MAP (Misconception Annotation Project) competition.

### Key Competition Assets
- **Adaptive Gating**: Use energy-aware activation for selective model inference
- **Runtime Monitoring**: Real-time performance tracking and adjustment
- **Configuration Presets**: Domain-optimized parameters for different datasets
- **Statistical Validation**: Bootstrap confidence intervals and performance metrics

## Common Development Commands

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test
uv run pytest tests/test_core.py

# Run with coverage
uv run pytest --cov=src/sundew

# Run SDK tests
uv run pytest tests/test_ipc*.py tests/test_grpc*.py -v
```

### Code Quality
```bash
# Format and lint (ruff is configured in pyproject.toml)
uv run ruff check src tests
uv run ruff format src tests

# Type checking
uv run mypy src

# Run all quality checks together
uv run ruff check src tests && uv run ruff format src tests && uv run mypy src
```

### Development Workflow
```bash
# Watch tests during development
uv run pytest --watch

# Run specific test with verbose output
uv run pytest tests/test_core.py::test_specific_function -v

# Generate coverage report
uv run pytest --cov=src/sundew --cov-report=html

# Profile algorithm performance
uv run python -m cProfile -o profile.stats benchmarks/run_dataset_suite.py

# Debug with pdb
uv run python -m pdb benchmarks/plot_single_run.py
```

### Running Benchmarks and Demos
```bash
# Full evidence suite (all benchmarks, ablations, bootstrap metrics)
uv run python tools/run_full_evidence.py

# Dataset suite with multiple presets
uv run python benchmarks/run_dataset_suite.py --presets tuned_v2 auto_tuned aggressive

# Single run analysis
uv run python benchmarks/plot_single_run.py --preset tuned_v2 --events 400

# CLI demo
uv run python -m sundew.cli --demo
uv run python -m sundew.cli --events 100 --temperature 0.1
```

### Key Benchmark Scripts
- `benchmarks/run_dataset_suite.py` - Run algorithm against multiple datasets with different presets
- `benchmarks/run_pipeline_dataset.py` - Process specific datasets with runtime telemetry logging
- `benchmarks/plot_layered_precision.py` - Generate layered classifier precision plots
- `benchmarks/run_ablation_study.py` - Ablation studies for algorithm components
- `benchmarks/run_adversarial_stream.py` - Test against adversarial input patterns

### SDK Development Commands
```bash
# Generate IPC bindings (requires grpcio-tools)
uv run python tools/generate_ipc_bindings.py

# Run SDK demo
uv run python examples/ipc_demo.py

# Send test events to IPC server
uv run python tools/send_score_event.py --port 8765 --feature glucose_mgdl=140

# Capture power measurements (requires INA219 sensor)
uv run python benchmarks/power/capture_power.py --duration 60 --output power.csv
```

## Code Architecture

### Core Components

**Main Algorithm (`src/sundew/core.py`)**
- `SundewAlgorithm` - Main algorithm class with PI control, energy pressure, gating
- `ProcessingResult` - Result container for processed events
- `Metrics` - Comprehensive metrics tracking including energy, activation rates, telemetry

**Configuration (`src/sundew/config.py`, `src/sundew/config_presets.py`)**
- `SundewConfig` - Configuration dataclass with validation
- Presets include: `tuned_v2`, `auto_tuned`, `aggressive`, `conservative`, `energy_saver`
- Special presets: `custom_health_hd82`, `custom_breast_probe` (probe-aware for medical data)

**Runtime System (`src/sundew/runtime.py`)**
- `PipelineRuntime` - Modern modular pipeline for composing significance, control, gating, energy stages
- `build_legacy_runtime()` - Bridges to original `SundewAlgorithm` for backward compatibility
- `build_simple_runtime()` - Lightweight pipeline for basic use cases
- Runtime adapters in `src/sundew/runtime_adapters.py` provide seamless interface bridging

**Legacy vs New Architecture**
- **Legacy**: `SundewAlgorithm` in `core.py` - Original monolithic implementation
- **New**: `PipelineRuntime` with pluggable components via interfaces in `interfaces.py`
- **Migration**: Use `build_legacy_runtime()` to gradually transition existing code
- **Recommendation**: Use `PipelineRuntime` for new implementations, maintain legacy for compatibility

**Energy Management (`src/sundew/energy.py`)**
- `EnergyAccount` - Energy tracking with regeneration and consumption
- Energy models with cap-aware management and AIMD control

**Gating & Control (`src/sundew/gating.py`)**
- Probabilistic gating with hysteresis
- Temperature-based activation decisions
- Gate probability calculations

### Key Interfaces (`src/sundew/interfaces.py`)
- `SignificanceModel` - For computing event significance
- `ControlPolicy` - For threshold adaptation
- `GatingStrategy` - For activation decisions
- `EnergyModel` - For energy accounting

### Data Processing Pipeline
1. **Event Input** → **Significance Calculation** (weighted combination of magnitude, anomaly, context, urgency)
2. **Control Policy** → **Threshold Adaptation** (PI controller with energy pressure)
3. **Gating Strategy** → **Activation Decision** (probabilistic with hysteresis)
4. **Energy Management** → **Resource Accounting** (consumption, regeneration, caps)

### Sundew Core SDK (`src/sundew_core_sdk/`)

The SDK layer enables hardware deployment on embedded devices (Jetson Nano, Coral Edge TPU, Raspberry Pi).

**SDK Core Components:**
- `SDKConfig` - Lightweight configuration for embedded deployment
- `AdaptiveGateController` - Facade wrapping `SundewAlgorithm` for firmware integration
- `MetricsTracker` - Rolling window metrics collection (activation rate, power, energy)
- `TelemetryEvent` - Telemetry data structures for firmware reporting

**IPC Layer (`src/sundew_core_sdk/ipc/`):**
- `IPCAdapter` - Bridges protobuf messages with SDK controller
- `IPCServer` - TCP/Unix socket transport for firmware communication
- `grpc_transport` - Production gRPC service for streaming gate decisions
- `bindings` - Protobuf/gRPC module loaders with version validation
- `shim` - C-facing interface for firmware integration

**Protocol Definition:**
- `docs/sdk/ipc/sundew_ipc_v1.proto` - Protobuf schema defining ScoreEvent, GateDecision, TelemetryPush messages
- Generated bindings: `src/sundew_ipc_v1_pb2.py`, `src/sundew_ipc_v1_pb2_grpc.py`

**Hardware Support:**
- `hardware/registry.py` - Board adapter registry (Jetson, Coral, RPi placeholders)
- `firmware/interface.py` - C ABI specification (future implementation)

**SDK Workflow:**
1. Firmware sends `ScoreEvent` (features) via IPC → `IPCAdapter`
2. `AdaptiveGateController` decides activation → returns `GateDecision`
3. Metrics tracked in `MetricsTracker` → exported as `TelemetryEvent`
4. Power measurements via INA219/INA3221 sensors correlated with telemetry

See `docs/sdk/README.md` for complete SDK documentation and examples.

## Testing Strategy

Tests are organized by component:
- `test_core.py` - Core algorithm functionality
- `test_config*.py` - Configuration validation and presets
- `test_gating*.py` - Gating strategy tests including edge cases
- `test_energy.py` - Energy model validation
- `test_cli*.py` - CLI interface tests
- `test_pipeline_runtime.py` - Runtime system tests
- `test_ipc*.py` - SDK IPC layer tests (adapter, shim, transport, bindings)
- `test_grpc*.py` - gRPC transport tests

Use `hypothesis` for property-based testing of algorithm invariants.

### Test Organization Patterns
- **Component Tests**: Test individual modules (energy, gating, control)
- **Integration Tests**: Test full pipeline workflows
- **Property Tests**: Use `hypothesis` for invariant validation
- **Benchmark Tests**: Performance regression detection
- **Edge Case Tests**: Boundary conditions and error handling

## Data Organization

**Input Data (`data/raw/`)**
- CSV files for various domains: healthcare, IoT, ECG, network security
- Key datasets: `breast_cancer_wisconsin_enriched.csv`, `uci_heart_disease.csv`, `MIT-BIH Arrhythmia Database.csv`

**Results (`data/results/`)**
- Benchmark outputs: CSVs, JSON telemetry files
- Ablation studies: `ablation_study.csv`, `ablation_runs/`
- Adversarial tests: `adversarial_runs/`
- Bootstrap metrics: `bootstrap_summary.json`
- Layered precision: `layered_precision*.csv`

**Plots and Visualizations (`assets/`, `results/plots/`)**
- Performance charts, energy savings, threshold adaptation plots
- Key visualization: `assets/layered_precision.png`

## Hardware Integration

The project includes hardware validation infrastructure:
- `tools/power_capture_template.py` - Power measurement interface (implement `read_power_sample()`)
- `tools/merge_runtime_power.py` - Merge runtime telemetry with power measurements
- `tools/runtime_monitor.py` - Runtime event monitoring and alerting

## Key Presets for Different Use Cases

- **`tuned_v2`** - Balanced performance, good starting point
- **`auto_tuned`** - Dataset-adaptive, general streaming baseline
- **`custom_health_hd82`** - Heart disease optimized (82% energy savings, ~0.196 recall)
- **`custom_breast_probe`** - Breast cancer with enriched features (77% savings, ~0.118 recall)
- **`aggressive`** - High activation rate, lower energy savings
- **`conservative`** - Low activation rate, maximum energy savings
- **`energy_saver`** - Minimal processing, maximum efficiency

## Monitoring and Telemetry

Use `PipelineRuntime.add_listener(callback)` for event-level monitoring:
```python
def my_callback(event_id, context, result):
    # Log activation decisions, energy state, etc.
    pass

runtime.add_listener(my_callback)
```

See `docs/RUNTIME_MONITORING.md` for alert guidance and monitoring patterns.

## Performance Optimization Guide

### Preset Selection Decision Tree
1. **Healthcare/Medical Data**: Start with `custom_health_hd82` or `custom_breast_probe`
2. **IoT/Sensor Data**: Use `auto_tuned` for adaptive thresholds
3. **High Throughput Needed**: Try `aggressive` preset
4. **Maximum Energy Savings**: Use `conservative` or `energy_saver`
5. **Unknown Domain**: Start with `tuned_v2` as baseline

### Tuning Guidelines
- **activation_threshold**: Lower = more activations, higher energy usage
- **energy_pressure**: Controls energy-performance tradeoff
- **gate_temperature**: Higher = more exploration, lower = more exploitation
- **target_activation_rate**: Desired percentage of events to process

### Performance Debugging
```bash
# Profile algorithm bottlenecks
uv run python -m cProfile benchmarks/run_dataset_suite.py

# Memory usage analysis
uv run python -m memory_profiler benchmarks/run_pipeline_dataset.py

# Threshold adaptation visualization
uv run python benchmarks/plot_single_run.py --preset tuned_v2 --events 1000
```

## Troubleshooting

### Common Issues

**"No activations" or very low activation rate**
- Check `activation_threshold` - may be too high
- Verify significance calculation weights in config
- Ensure input data is properly normalized

**Poor energy savings**
- Increase `energy_pressure` parameter
- Use more conservative preset
- Check if `target_activation_rate` is too high

**Unstable threshold adaptation**
- Reduce PI controller gains (`kp`, `ki`)
- Increase `threshold_smoothing` factor
- Check for outliers in input data

**Import/Module errors**
- Ensure you're using `uv run` to activate virtual environment
- Check `PYTHONPATH` includes `src` directory
- Verify all dependencies installed: `uv pip list`

### Debugging Checklist
1. Run `uv run pytest tests/test_core.py` to verify core functionality
2. Check configuration with `uv run python -c "from sundew.config_presets import get_preset; print(get_preset('tuned_v2'))"`
3. Validate data format matches expected input schema
4. Enable verbose logging in algorithm initialization
5. Use `benchmarks/plot_single_run.py` for visual debugging

## Integration Patterns

### Adding New Algorithms
1. Implement interfaces in `src/sundew/interfaces.py`
2. Create component in appropriate module (gating, energy, etc.)
3. Add configuration parameters to `config.py`
4. Create preset in `config_presets.py`
5. Add tests following existing patterns
6. Benchmark against existing presets

### Competition Integration
```python
# Example: Integrating with ML pipeline
from sundew.runtime import build_simple_runtime
from sundew.config_presets import get_preset

# Create adaptive inference system
runtime = build_simple_runtime(get_preset('auto_tuned'))

def adaptive_predict(features):
    # Use Sundew to gate expensive model calls
    context = create_processing_context(features)
    result = runtime.process(context)

    if result.activated:
        return expensive_model(features)
    else:
        return cheap_baseline(features)
```

### Hardware Deployment
- Use `tools/power_capture_template.py` for real power measurement
- Implement `read_power_sample()` for your hardware platform
- Run `tools/merge_runtime_power.py` to correlate telemetry with power data
- Monitor with `tools/runtime_monitor.py` for production alerts

## Important Notes

- Algorithm performance is highly dependent on preset selection for the target domain
- Energy pressure and gating parameters require careful tuning for new datasets
- Bootstrap confidence intervals provide statistical validation for metrics
- Layered classifier (optional) can boost precision to ~1.0 while preserving recall
- Hardware validation workflow connects simulation with real device measurements
- Use `PipelineRuntime` for new code, `SundewAlgorithm` for legacy compatibility
- Always run full test suite before committing changes: `uv run pytest --cov=src/sundew`
