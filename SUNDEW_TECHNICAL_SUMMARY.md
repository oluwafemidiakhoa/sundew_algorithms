# Sundew: A Simple Energy-Aware Gating Algorithm

## What It Does

Sundew is an algorithm that decides whether to process incoming data based on its estimated importance. Think of it as a smart filter that only activates expensive computation when the input seems significant enough to justify the energy cost.

## Core Concept

The algorithm works in three steps:

1. **Evaluate significance**: For each input, compute a significance score (0-1) based on features like magnitude, anomaly level, and context
2. **Make gating decision**: Compare significance to an adaptive threshold to decide whether to process
3. **Adapt threshold**: Adjust the threshold up or down to maintain a target activation rate

## Key Innovation

The main contribution is using a PI controller with hysteresis to maintain stable activation rates while adapting to changing input patterns. This prevents the system from oscillating between over- and under-processing.

## Technical Details

### Significance Function
```
significance = w1*magnitude + w2*anomaly + w3*context + w4*urgency
```
Where weights sum to 1 and features are normalized to [0,1].

### Threshold Adaptation
```
threshold_new = threshold_old + Kp*error + Ki*integral_error - energy_pressure
```
- error = target_rate - observed_rate
- energy_pressure increases threshold when energy is low

### Gating with Hysteresis
```
if previous_state == active:
    threshold_effective = threshold - gap
else:
    threshold_effective = threshold + gap

activate = (significance > threshold_effective)
```

## Performance

In testing on time-series data:
- Reduces processing by 70-85% (processes only 15-30% of inputs)
- Maintains 90-95% of baseline accuracy on anomaly detection tasks
- Adds <0.5ms overhead per decision

## Limitations

- Requires tuning for different data distributions
- Initial learning period needed to establish baselines
- Not suitable for applications where every input is critical
- Performance depends heavily on significance function quality

## Use Cases

Works well for:
- Sensor monitoring with rare interesting events
- Video processing where most frames are similar
- IoT devices with limited battery life
- Any streaming data with low information density

Not recommended for:
- Safety-critical systems requiring 100% coverage
- Data with uniformly high importance
- Applications with strict latency requirements

## Implementation

The Python implementation (~500 lines) provides:
- Basic algorithm with configurable parameters
- Energy tracking simulation
- Simple CLI for testing
- Presets for common scenarios

## Reproducibility

All results can be reproduced using the provided code and test data. The algorithm is deterministic when seeded, making benchmarks repeatable.

---

This is a research prototype exploring energy-aware computing. It's not production-ready and hasn't been validated for medical or safety-critical applications.
