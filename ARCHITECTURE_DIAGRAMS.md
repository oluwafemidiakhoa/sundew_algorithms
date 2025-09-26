# Sundew Algorithms - Architecture Diagrams

This document provides comprehensive visual documentation of the Sundew Algorithms application architecture, data flow, and performance characteristics.

## Diagram Overview

### 1. System Architecture Diagram
**File:** `assets/sundew_system_architecture.png`

Shows the complete system architecture with:
- **Core Algorithm Layer**: SundewAlgorithm with PI Controller, Energy Pressure, and Gating Logic
- **Configuration Layer**: SundewConfig with preset management
- **Runtime Pipeline Layer**: Modular pipeline framework
- **Interface Layer**: SignificanceModel, ControlPolicy, GatingStrategy, EnergyModel
- **Data Layer**: 6 datasets (Breast Cancer, Heart Disease, IoT, ECG, Financial, Network Security)
- **Benchmarking Layer**: Dataset Suite, Ablation Studies, Bootstrap Metrics, Adversarial Testing
- **Tools Layer**: Power Capture, Runtime Monitor, Plotting, CLI Interface
- **Results Layer**: CSV, JSON, PNG outputs with statistical validation

### 2. Algorithm Pipeline Diagram
**File:** `assets/sundew_algorithm_pipeline.png`

Detailed processing pipeline showing:
- **Event Input**: Raw sensor data and time series events
- **Significance Calculation**: Weighted combination of magnitude, anomaly, context, urgency
- **PI Controller & Threshold**: Adaptive threshold control with error feedback
- **Energy Accounting**: Consumption, regeneration, cap-aware management
- **Gating Decision**: Probabilistic activation using sigmoid with temperature
- **Process/Dormant Modes**: Event processing vs. energy regeneration
- **Feedback Loops**: Rate control and energy management feedback
- **Key Metrics**: Activation rates, energy savings, F1/recall scores, confidence intervals
- **Configuration Presets**: Domain-optimized parameter sets

### 3. Data Flow Diagram
**File:** `assets/sundew_data_flow.png`

Complete data processing workflow:
- **Input Sources**: 6 datasets with sample counts
- **Multi-Preset Processing**: 10+ presets applied to each dataset
- **Analysis Branches**:
  - Dataset Suite Benchmarking
  - Ablation Studies
  - Bootstrap Metrics (statistical validation)
  - Layered Classifier (precision enhancement)
  - Adversarial Testing (robustness)
- **Results Outputs**: CSV results, JSON telemetry, PNG plots, statistical CIs, reports
- **Storage Locations**: Organized file structure in data/results/, assets/, docs/

### 4. Performance Summary Dashboard
**File:** `assets/sundew_performance_dashboard.png`

Comprehensive performance visualization:
- **Recall vs Energy Savings Scatter**: Shows trade-offs by dataset
- **Average Energy Savings by Preset**: Bar chart comparing preset efficiency
- **Precision Improvement with Layered Classifier**: Before/after comparison
- **Bootstrap Confidence Intervals**: Statistical validation (95% CIs)

## Key Architecture Patterns

### Modular Pipeline Design
The application uses a modular pipeline architecture where:
- **Interfaces** define contracts for pluggable components
- **Runtime adapters** bridge legacy and new implementations
- **Configuration presets** provide domain-optimized parameter sets
- **Metrics collection** provides comprehensive telemetry

### Data Processing Flow
1. **Input** → Raw datasets (CSV format)
2. **Configuration** → Preset selection and parameter tuning
3. **Processing** → Algorithm execution with multiple presets
4. **Analysis** → Statistical validation and performance measurement
5. **Output** → Results in multiple formats (CSV, JSON, PNG, SVG)

### Quality Assurance Pipeline
- **Unit Tests** → Core algorithm validation
- **Integration Tests** → Pipeline runtime testing
- **Property-Based Tests** → Algorithm invariants (using hypothesis)
- **Benchmark Tests** → Performance regression detection
- **Statistical Validation** → Bootstrap confidence intervals

## Performance Characteristics

### Energy Efficiency
- **Conservative presets**: 92-94% energy savings
- **Balanced presets**: 82-90% energy savings
- **Aggressive presets**: 80-87% energy savings

### Domain Optimization
- **Heart Disease (custom_health_hd82)**: 82% savings, 0.196 recall
- **Breast Cancer (custom_breast_probe)**: 77% savings, 0.118 recall
- **IoT Sensors (auto_tuned)**: 88% savings, 0.500 recall

### Precision Enhancement
- **Layered Classifier**: Improves precision from 0.22-0.67 to 0.90-1.00
- **Bootstrap Validation**: 95% confidence intervals for statistical certainty
- **Ablation Studies**: Component-wise performance analysis

## Implementation Notes

### Technology Stack
- **Language**: Python 3.10+
- **Package Manager**: uv
- **Core Dependencies**: numpy, pandas
- **Development Tools**: ruff, mypy, pytest, hypothesis
- **Visualization**: matplotlib for diagrams and plots

### File Organization
```
assets/                          # Diagrams and visualizations
├── sundew_system_architecture.*  # System architecture
├── sundew_algorithm_pipeline.*   # Processing pipeline
├── sundew_data_flow.*            # Data flow and results
└── sundew_performance_dashboard.* # Performance summary

data/results/                    # Comprehensive results
├── dataset_suite_full.csv       # Main benchmark results
├── dataset_runs_full/           # Detailed JSON logs
├── bootstrap_summary_extended.json # Statistical validation
├── layered_precision_extended.csv # Precision enhancement
└── ablation_study_extended.csv  # Component analysis

src/sundew/                      # Core implementation
├── core.py                      # Main algorithm
├── config.py                    # Configuration management
├── runtime.py                   # Pipeline framework
└── interfaces.py                # Component contracts
```

### Hardware Integration
The architecture supports hardware validation through:
- **Power capture templates** for real-world energy measurement
- **Runtime telemetry** for device monitoring
- **Merge utilities** for combining simulation and hardware data

## Usage Examples

### Running Complete Analysis
```bash
# Full evidence suite (all datasets, presets, analyses)
uv run python tools/run_full_evidence.py

# Specific dataset with custom preset
uv run python benchmarks/run_pipeline_dataset.py data/raw/heart_disease.csv --preset custom_health_hd82

# Generate layered precision analysis
uv run python benchmarks/layer_classifier.py <dataset_logs> --out results.csv
```

### Creating Custom Diagrams
```bash
# Generate all architecture diagrams
uv run python create_architecture_diagram.py

# Individual plot generation
uv run python benchmarks/plot_layered_precision.py --out assets/precision.png
```

This visual documentation provides a complete understanding of the Sundew Algorithms architecture, enabling developers to quickly grasp the system design, data flow, and performance characteristics.
