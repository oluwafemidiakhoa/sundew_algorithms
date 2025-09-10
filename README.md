# Sundew Algorithm
**Energy-Aware Selective Activation for Edge AI Systems**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> 🌿 *"Nature's wisdom, encoded in silicon."*

## What is Sundew?

Sundew is a **bio-inspired, lightweight algorithm** that dramatically reduces energy consumption in edge AI systems by staying dormant until truly significant events occur. Like the carnivorous sundew plant that conserves energy by waiting for the perfect moment to capture prey, this algorithm keeps expensive AI computations asleep until they're genuinely needed.

**Core Innovation**: Event-driven intelligence that achieves **83-90% energy savings** through intelligent selective activation.

### Key Principles

- **Significance Scoring**: Events are evaluated for importance using weighted feature combinations
- **Temperature-Controlled Gating**: Adaptive thresholds determine when to "wake up" expensive processing
- **Closed-Loop Control**: PI controller automatically adjusts activation rates while respecting energy budgets
- **Minimal Dependencies**: Lightweight, stdlib-only core designed for constrained edge devices

### Why It Matters

Always-on AI inference is a major energy drain on battery-powered devices. Sundew's dormant-until-useful behavior has demonstrated consistent **83-90% energy savings** across synthetic and real-world datasets (including ECG arrhythmia detection), making it ideal for:

- Wearable health monitors
- Edge security cameras
- Autonomous robotics
- Space exploration systems
- IoT sensor networks

## Installation

### Quick Install
```bash
# Core library (lightweight, no heavy dependencies)
pip install sundew-algorithms

# With optional benchmarking and visualization tools
pip install "sundew-algorithms[benchmarks]"
```

### Development Setup
```bash
# Create and activate virtual environment
python -m venv .venv

# Windows Command Prompt
.venv\Scripts\activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

# Install with development dependencies
pip install "sundew-algorithms[benchmarks]"
```

**Compatibility**: Python 3.10+ (tested on 3.10-3.13)

## Quick Start

### Command Line Interface

```bash
# Basic help
sundew --help
# or
python -m sundew --help

# Run synthetic demonstration
sundew --demo --events 200 --temperature 0.1
```

The demo outputs real-time energy, threshold, and activation statistics, giving you an immediate sense of how Sundew adapts to changing conditions.

### Programmatic Usage

```python
from sundew.config_presets import get_preset
from sundew.core import SundewAlgorithm

# Load a proven configuration
cfg = get_preset("tuned_v2")        # Balanced default
# cfg = get_preset("ecg_mitbih_best") # Optimized for ECG data

# Initialize the algorithm
algo = SundewAlgorithm(cfg)

# Process your event stream
for event in event_stream:
    # event should be a dict with features normalized to [0,1]
    # (except 'magnitude' which can be 0-100)
    result = algo.process(event)
    
    if result is not None:
        # Significant event detected - run expensive AI processing
        handle_critical_event(result)
    # else: system stays dormant, saving energy

# Get comprehensive performance report
report = algo.report()
print(f"Energy savings: {report.energy_savings:.1f}%")
print(f"Activation rate: {report.activation_rate:.2%}")
```

## Algorithm Overview

Sundew implements a sophisticated yet lightweight control system with four core components:

### 1. Significance Scoring
```
s ∈ [0,1] = Σᵢ wᵢ fᵢ(x)
```
Events are scored using weighted feature combinations (magnitude, anomaly, context, urgency).

### 2. Temperature Gating
```
p = σ((s - θ) / τ)
```
- **τ → 0**: Hard deterministic gate (production mode)
- **τ > 0**: Soft probabilistic gate (analysis mode)

### 3. Adaptive Threshold Control
```
θ ← clip(θ + kₚe + kᵢΣe + λ(1 - E/Eₘₐₓ), θₘᵢₙ, θₘₐₓ)
```
PI controller with energy pressure: **e = p* - p̂** (target vs actual activation rate)

### 4. Energy Accounting
Compares baseline (always-on) versus actual energy consumption to provide realistic savings estimates.

## Real-World Example: ECG Arrhythmia Detection

### Dataset Preparation
Place your ECG data in the `data/` directory:
```
data/MIT-BIH Arrhythmia Database.csv
```

### Single Experiment
```bash
# Windows
python -m benchmarks.run_ecg ^
  --csv "data\MIT-BIH Arrhythmia Database.csv" ^
  --preset tuned_v2 ^
  --limit 50000 ^
  --save "results\ecg_experiment.json"

# macOS/Linux
python -m benchmarks.run_ecg \
  --csv "data/MIT-BIH Arrhythmia Database.csv" \
  --preset tuned_v2 \
  --limit 50000 \
  --save "results/ecg_experiment.json"
```

**Example Results** (50k samples):
```
Precision: 0.144    Recall: 0.179    F1: 0.160
Energy Savings: ~83%    Activation Rate: ~15%
```

### Performance Tuning
```bash
# Optimize for higher precision (fewer false alarms)
python -m benchmarks.run_ecg \
  --csv "data/MIT-BIH Arrhythmia Database.csv" \
  --preset tuned_v2 \
  --overrides "target_activation_rate=0.10,gate_temperature=0.05" \
  --save "results/high_precision_run.json"

# Add refractory period to prevent rapid retriggering
--refractory 5
```

### Hyperparameter Optimization

#### 1. Parameter Sweep
```bash
python -m benchmarks.sweep_ecg \
  --csv "data/MIT-BIH Arrhythmia Database.csv" \
  --preset ecg_v1 \
  --limit 50000 \
  --out "results/parameter_sweep.csv"
```

#### 2. Select Optimal Configurations
```bash
python -m benchmarks.select_best \
  --csv "results/parameter_sweep.csv" \
  --out-csv "results/best_configs.csv" \
  --out-md "results/best_configs.md" \
  --min-savings 88 \
  --sort f1,precision \
  --top-n 20 --describe
```

#### 3. Deploy Best Configuration
```bash
python -m benchmarks.run_ecg \
  --csv "data/MIT-BIH Arrhythmia Database.csv" \
  --preset ecg_mitbih_best \  # Frozen optimal config
  --limit 50000 \
  --save "results/production_run.json"
```

### Analysis and Evaluation
```bash
# Detailed performance analysis
python -m benchmarks.eval_classification --json "results/ecg_experiment.json"

# Generate visualizations
python -m benchmarks.plot_single_run --json "results/ecg_experiment.json" --out "results/"
python -m benchmarks.plot_best_tradeoffs --csv "results/best_configs.csv" --out "results/tradeoffs.png"
```

## Configuration Presets

Sundew includes battle-tested presets for different use cases:

| Preset | Use Case | Characteristics |
|--------|----------|----------------|
| `tuned_v2` | **General purpose** | Balanced PI control with energy pressure |
| `ecg_v1` | **ECG monitoring** | Wider gate, lower threshold for arrhythmia recall |
| `ecg_mitbih_best` | **Production ECG** | Frozen optimal configuration from hyperparameter sweep |
| `aggressive` | **Maximum savings** | Higher thresholds, faster adaptation |
| `conservative` | **High precision** | Stricter activation criteria |
| `energy_saver` | **Battery critical** | Maximum energy conservation |
| `high_temp` | **Exploration** | Softer gating for analysis |
| `low_temp` | **Production** | Hard deterministic gating |

```python
# List all available presets
from sundew.config_presets import list_presets
print(list_presets())

# Load and inspect a preset
cfg = get_preset("ecg_mitbih_best")
print(cfg)  # Shows all parameter values
```

## Application Domains

| Domain | Applications | Benefits |
|--------|-------------|----------|
| **Healthcare** | Wearable monitors, arrhythmia detection, patient triage | Extended battery life, continuous monitoring |
| **Security** | Smart cameras, acoustic triggers, intrusion detection | Reduced bandwidth, privacy-preserving |
| **Robotics** | SLAM updates, obstacle detection, duty-cycled perception | Longer missions, thermal management |
| **Space Systems** | Planetary rovers, satellite monitoring, deep space probes | Critical for power-constrained missions |
| **Neuromorphic** | Event-driven processing, spiking neural networks | Natural fit for asynchronous architectures |
| **IoT Sensors** | Environmental monitoring, predictive maintenance | Years of battery life vs. days |

## Repository Structure

```
sundew_algorithms/
├── src/sundew/                 # Core algorithm library
│   ├── __init__.py            # Main entry point
│   ├── cli.py                 # Command-line interface
│   ├── config.py              # Configuration dataclass
│   ├── config_presets.py      # Pre-tuned configurations
│   ├── core.py                # Algorithm implementation
│   ├── energy.py              # Energy accounting model
│   └── gating.py              # Temperature-controlled gating
│
├── benchmarks/                 # Evaluation and analysis tools
│   ├── run_ecg.py             # ECG dataset experiments
│   ├── eval_classification.py # Performance metrics
│   ├── sweep_ecg.py           # Hyperparameter optimization
│   ├── select_best.py         # Configuration selection
│   ├── plot_*.py              # Visualization tools
│   └── synthetic_stream.py    # Synthetic data generation
│
├── tests/                      # Comprehensive test suite
├── data/                       # Input datasets (e.g., MIT-BIH)
├── results/                    # Output files (JSON, CSV, plots)
└── docs/                       # Additional documentation
```

## Development and Testing

### Running Tests
```bash
# Basic test suite
pytest -v

# With coverage reporting
pytest --cov=src --cov-report=term-missing

# Test specific modules
pytest tests/test_core.py -v
```

### Code Quality
```bash
# Linting and formatting
ruff check .      # Check for issues
ruff format .     # Auto-format code

# Type checking (if mypy is installed)
mypy src/
```

### Pre-Commit Checklist
Before submitting PRs:
```bash
ruff check . && ruff format .
pytest --cov=src --cov-report=term-missing
```

## Platform Notes

- **Windows**: Emoji display may vary by console; CLI auto-downgrades to ASCII if UTF-8 isn't supported
- **Memory**: Minimal footprint suitable for embedded systems
- **Dependencies**: Core library uses only Python stdlib; benchmarks add NumPy/Matplotlib

## Contributing

We welcome contributions! High-impact areas include:

### Priority Areas
- **Domain-specific feature engineering** (healthcare, security, robotics)
- **Advanced control algorithms** (PID, adaptive gains, model-predictive control)
- **Device-calibrated energy models** with public benchmarks
- **Visualization and evaluation tools**
- **Documentation and tutorials**

### Contribution Process
1. Fork the repository
2. Create a feature branch
3. Implement your changes with tests
4. Run the pre-commit checklist
5. Submit a pull request with clear description

## Citation

If you use Sundew in your research, please cite:

```bibtex
@techreport{Idiakhoa2025Sundew,
  title  = {Sundew Algorithm: Energy-Aware Selective Activation for Edge AI},
  author = {Oluwafemi Idiakhoa},
  year   = {2025},
  note   = {Open-source implementation with real-data validation on MIT-BIH ECG},
  url    = {https://github.com/oluwafemidiakhoa/sundew_algorithms}
}
```

## Release History

### v0.1.9 (Latest)
- ✅ Fixed PyPI/TestPyPI publishing pipeline
- ✅ Enhanced CLI: `python -m sundew` now works reliably
- ✅ Refined metadata compliance (PEP 639)
- ✅ Pre-built wheels for Python 3.10-3.13
- ✅ Maintained lightweight core with optional benchmark dependencies

### Previous Versions
- **v0.1.8**: Initial algorithm implementation
- **v0.1.7**: ECG benchmark integration
- **v0.1.6**: Configuration preset system

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: This README and inline docstrings
- **Issues**: GitHub issue tracker for bugs and feature requests
- **Discussions**: GitHub discussions for questions and community support

---

*Sundew: Making edge AI sustainable, one activation at a time.* 🌿