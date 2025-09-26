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

## Common Development Commands

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test
uv run pytest tests/test_core.py

# Run with coverage
uv run pytest --cov=src/sundew
```

### Code Quality
```bash
# Format and lint (ruff is configured in pyproject.toml)
uv run ruff check src tests
uv run ruff format src tests

# Type checking
uv run mypy src
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
- `PipelineRuntime` - Modular pipeline for composing significance, control, gating, energy stages
- `build_legacy_runtime()`, `build_simple_runtime()` - Factory functions
- Runtime adapters in `src/sundew/runtime_adapters.py` bridge old and new interfaces

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

## Testing Strategy

Tests are organized by component:
- `test_core.py` - Core algorithm functionality
- `test_config*.py` - Configuration validation and presets
- `test_gating*.py` - Gating strategy tests including edge cases
- `test_energy.py` - Energy model validation
- `test_cli*.py` - CLI interface tests
- `test_pipeline_runtime.py` - Runtime system tests

Use `hypothesis` for property-based testing of algorithm invariants.

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

## Important Notes

- Algorithm performance is highly dependent on preset selection for the target domain
- Energy pressure and gating parameters require careful tuning for new datasets
- Bootstrap confidence intervals provide statistical validation for metrics
- Layered classifier (optional) can boost precision to ~1.0 while preserving recall
- Hardware validation workflow connects simulation with real device measurements
