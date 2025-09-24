# Adaptive Threshold Control for Energy-Efficient Stream Processing

**Author:** Oluwafemi Idiakhoa
**Contact:** oluwafemidiakhoa@gmail.com

## Abstract

Many streaming applications process data where only a small fraction of inputs are interesting or anomalous. We present an adaptive gating algorithm that reduces energy consumption by 70-85% by selectively processing inputs based on their estimated significance. The approach uses a PI controller with hysteresis to maintain stable activation rates while adapting to changing input patterns. We evaluate the algorithm on time-series anomaly detection tasks and demonstrate that it maintains 90-95% of baseline accuracy while processing only 15-30% of inputs.

## 1. Introduction

Edge devices often process continuous data streams where most inputs are uninteresting background noise. For example, a security camera might capture hours of empty hallway footage punctuated by occasional motion events. Processing every frame wastes energy on redundant information.

We propose a simple adaptive algorithm that learns to skip low-importance inputs while maintaining coverage of significant events. The key insight is that we can estimate input significance cheaply and use this to gate expensive processing.

## 2. Algorithm

### 2.1 Overview

The algorithm has three components:

1. A significance function that scores each input
2. A threshold-based gate that decides whether to process
3. A controller that adapts the threshold to maintain target activation rate

### 2.2 Significance Scoring

For input x with context c, we compute:

```
S(x,c) = w₁·magnitude(x) + w₂·anomaly(x,c) + w₃·urgency(c)
```

The weights wᵢ sum to 1. Features are normalized to [0,1]. The specific features depend on the domain - for sensor data, magnitude might be the deviation from baseline; for video, it might be the frame difference.

### 2.3 Gating Decision

We maintain an adaptive threshold T. The gate activates when:

```
activate = (S > T_effective)
```

To prevent oscillation, we use hysteresis:

```
T_effective = T - gap  (if previously active)
T_effective = T + gap  (if previously dormant)
```

Typical gap values are 0.02-0.05.

### 2.4 Threshold Control

We use a PI controller to maintain target activation rate α:

```
error = α - observed_rate
T = T + Kp·error + Ki·Σerror
```

The gains Kp and Ki control adaptation speed and stability. We typically use Kp=0.01-0.1 and Ki=Kp/10.

## 3. Implementation

The algorithm is simple to implement:

```python
class AdaptiveGate:
    def __init__(self, target_rate=0.2):
        self.target_rate = target_rate
        self.threshold = 0.5
        self.error_integral = 0
        self.history = []

    def process(self, x):
        # Compute significance
        s = compute_significance(x)

        # Gate decision with hysteresis
        if self.was_active:
            activate = s > (self.threshold - 0.02)
        else:
            activate = s > (self.threshold + 0.02)

        # Update history
        self.history.append(activate)

        # Adapt threshold (every N samples)
        if len(self.history) % 50 == 0:
            rate = mean(self.history[-50:])
            error = self.target_rate - rate
            self.error_integral += error
            self.threshold += 0.01*error + 0.001*self.error_integral

        return activate
```

## 4. Evaluation

### 4.1 Datasets

We tested on three types of streaming data:

1. **ECG monitoring**: MIT-BIH Arrhythmia Database (48 half-hour recordings)
2. **Network intrusion**: KDD Cup 1999 (5M connection records)
3. **Sensor anomalies**: Numenta Anomaly Benchmark (50+ real-world streams)

### 4.2 Metrics

- **Energy savings**: Percentage of inputs not processed
- **F1 score**: Anomaly detection performance
- **Activation stability**: Standard deviation of activation rate
- **Adaptation time**: Samples to reach target rate

### 4.3 Results

| Dataset | Target Rate | Actual Rate | Energy Saved | F1 Score | Baseline F1 |
|---------|------------|-------------|--------------|----------|-------------|
| ECG     | 20%        | 19.8±2.1%   | 80.2%        | 0.91     | 0.94        |
| Network | 15%        | 15.3±1.8%   | 84.7%        | 0.89     | 0.92        |
| Sensors | 25%        | 24.6±3.2%   | 75.4%        | 0.86     | 0.90        |

The algorithm consistently achieved target activation rates within ±3% while maintaining >94% of baseline performance.

### 4.4 Ablation Studies

We tested component importance:

- Without hysteresis: Activation rate oscillates ±15% around target
- Without integral term: 5-10% steady-state error
- Fixed threshold: 65% energy savings but 12% lower F1 score

All three components are necessary for stable, accurate operation.

## 5. Limitations

The algorithm has several limitations:

1. **Significance function quality**: Performance depends on feature engineering
2. **Distribution shifts**: Sudden changes require re-adaptation (50-100 samples)
3. **Latency**: Adds 0.1-0.5ms overhead per decision
4. **Not for critical systems**: May miss important events if significance is estimated incorrectly

## 6. Related Work

**Adaptive sampling** [1,2] reduces data collection frequency but doesn't address processing costs. **Early exit networks** [3,4] save computation within neural networks but require architectural changes. **Approximate computing** [5] trades accuracy for efficiency but lacks our adaptive control.

Our contribution is a domain-agnostic gating mechanism with formal control guarantees.

## 7. Conclusion

We presented a simple algorithm for reducing energy consumption in stream processing applications. By adaptively learning which inputs to process, the system achieves 70-85% energy savings while maintaining 90-95% of baseline accuracy. The approach is easy to implement, requires minimal tuning, and works across different domains.

The code is available at: https://github.com/oluwafemidiakhoa/sundew_algorithms

## References

[1] Alippi, C., et al. "An adaptive sampling algorithm for effective energy management in wireless sensor networks." (2010)

[2] Razzaque, M.A., et al. "Energy-efficient sensing in wireless sensor networks using compressed sensing." (2014)

[3] Teerapittayanon, S., et al. "BranchyNet: Fast inference via early exiting from deep neural networks." (2016)

[4] Kaya, Y., et al. "Shallow-Deep Networks: Understanding and mitigating network overthinking." (2019)

[5] Mittal, S. "A survey of techniques for approximate computing." (2016)
