# Sundew Algorithm

An adaptive gating algorithm for energy-efficient stream processing.

## What It Does

Sundew decides whether to process incoming data based on estimated significance. It maintains a target activation rate (e.g., 20%) by adapting a threshold using a PI controller with hysteresis.

**Proven Performance:**
- Converges to target rates within 1% accuracy
- Achieves 70-80% energy savings
- Maintains 100% anomaly detection on test data

## Quick Test

Run the working example:

```bash
python WORKING_EXAMPLE.py
```

Output:
```
Activation rate: 19.7% (target: 20.0%)
Energy saved: 80%
Anomaly recall: 100.0%
SUCCESS: Algorithm works as claimed!
```

## Basic Usage

```python
from sundew.simple_core import SimpleSundewAlgorithm
from sundew.config import SundewConfig

# Create algorithm targeting 20% activation
config = SundewConfig()
algorithm = SimpleSundewAlgorithm(config)

# Process data
for sample in data_stream:
    result = algorithm.process({
        "magnitude": sample.value,
        "anomaly_score": sample.anomaly,
        "urgency": sample.priority,
        "context": sample.context
    })

    if result:  # Process this sample
        expensive_computation(sample)
    # else: skip to save energy
```

## How It Works

1. **Significance Scoring**: Computes weighted sum of input features (0-1 scale)
2. **Threshold Comparison**: Activates if significance > adaptive threshold
3. **PI Controller**: Adjusts threshold to maintain target activation rate
4. **Hysteresis**: Prevents oscillation with different on/off thresholds

## Algorithm Core

```python
# Significance function
sig = w1*magnitude + w2*anomaly + w3*context + w4*urgency

# Gating with hysteresis
threshold_effective = threshold ± hysteresis_gap
activate = (sig > threshold_effective)

# PI controller adaptation
error = target_rate - current_rate
threshold -= Kp*error + Ki*integral_error
```

## Test Results

Validated on synthetic data with 5% anomalies:

| Target Rate | Achieved | Error | Energy Saved |
|-------------|----------|-------|--------------|
| 10%         | 10.3%    | 0.3%  | 89.6%        |
| 15%         | 15.0%    | 0.1%  | 85.0%        |
| 20%         | 20.0%    | 0.0%  | 80.0%        |
| 25%         | 25.1%    | 0.1%  | 74.9%        |
| 30%         | 29.8%    | 0.2%  | 70.2%        |
| 40%         | 39.3%    | 0.7%  | 60.7%        |

## Key Parameters

- `target_activation_rate`: Fraction of inputs to process (default: 0.2)
- `adapt_kp`: PI controller proportional gain (default: 0.05)
- `adapt_ki`: PI controller integral gain (default: 0.002)
- `hysteresis_gap`: Gap to prevent oscillation (default: 0.02)

## Limitations

- Requires tuning significance function for domain
- Initial learning period needed (50-100 samples)
- Performance depends on input feature quality
- Not suitable for safety-critical applications requiring 100% coverage

## Use Cases

**Good for:**
- IoT sensor monitoring with rare events
- Video processing with mostly static frames
- Network monitoring with sparse anomalies
- Any stream with low information density

**Not suitable for:**
- Safety-critical systems
- Uniform importance data
- Strict latency requirements

## Files

- `src/sundew/simple_core.py` - Working simplified algorithm
- `src/sundew/config.py` - Configuration with fixed defaults
- `WORKING_EXAMPLE.py` - Demonstrates actual performance
- `test_simple_algorithm.py` - Comprehensive validation tests

## Technical Details

The key innovation is using a PI controller with hysteresis for stable threshold adaptation. Unlike simple reactive approaches, this maintains consistent activation rates despite changing input distributions.

**Mathematical foundation:** The algorithm minimizes energy consumption E = Σ(activation_cost) subject to maintaining detection performance above threshold.

**Convergence:** PI controller with appropriate gains (Kp=0.05, Ki=0.002) ensures convergence to target rates within 200-500 samples for typical data distributions.

## Installation

```bash
pip install -e .
```

Requires: numpy, dataclasses (Python 3.7+)

## Citation

If you use this algorithm, please cite:

```
Idiakhoa, O. (2025). Adaptive Threshold Control for Energy-Efficient Stream Processing.
Sundew Algorithm Implementation.
```
