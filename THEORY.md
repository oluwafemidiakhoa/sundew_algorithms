# Sundew Algorithm: Mathematical Foundation

## Problem Statement

Given a stream of inputs X = {x₁, x₂, ..., xₙ}, we want to:
- Process only a fraction α (target rate) of inputs
- Maximize the importance of processed inputs
- Minimize energy consumption

## Core Algorithm

### 1. Significance Computation

For each input xᵢ, compute significance S(xᵢ) as a weighted sum:

```
S(xᵢ) = Σⱼ wⱼ · fⱼ(xᵢ)
```

Where:
- fⱼ(xᵢ) are feature functions extracting different aspects
- wⱼ are weights with Σwⱼ = 1
- S(xᵢ) ∈ [0, 1]

Example features:
- f₁: Magnitude (normalized deviation from baseline)
- f₂: Anomaly score (statistical outlier detection)
- f₃: Context relevance (domain-specific importance)
- f₄: Urgency (time-sensitive factors)

### 2. Gating Decision

Use a threshold T to decide activation:

```
activate(xᵢ) = {
    1  if S(xᵢ) > T
    0  otherwise
}
```

### 3. Threshold Adaptation

The key innovation is adaptive threshold control using a PI controller:

```
T(t+1) = T(t) + ΔT
ΔT = Kₚ · e(t) + Kᵢ · Σe(τ)
```

Where:
- e(t) = α - r(t) is the error (target rate - observed rate)
- Kₚ is proportional gain (typically 0.01-0.1)
- Kᵢ is integral gain (typically 0.001-0.01)
- r(t) is observed activation rate over recent window

### 4. Hysteresis for Stability

To prevent oscillation around the threshold:

```
T_effective = {
    T - h  if previously activated
    T + h  if previously dormant
}
```

Where h is the hysteresis gap (typically 0.01-0.05).

## Theoretical Analysis

### Convergence

Under mild assumptions (bounded significance distribution, proper gain tuning), the PI controller converges to the target rate:

```
lim(t→∞) |r(t) - α| < ε
```

The convergence rate depends on:
- Gain parameters Kₚ and Kᵢ
- Input distribution stability
- Window size for rate calculation

### Energy Model

Expected energy consumption:

```
E[Energy] = α · E_process + (1-α) · E_eval
```

Where:
- E_process >> E_eval (processing cost >> evaluation cost)
- Energy savings ≈ (1-α) · (E_process - E_eval) / E_process

For typical values (α=0.2, E_process=100, E_eval=1):
- Energy savings ≈ 79%

### Performance Bounds

Given perfect significance scoring:
- Best case: Process only true positives
- Worst case: Random selection
- Our approach: Between random and optimal

Expected performance:

```
P(sundew) = P(baseline) · (α + (1-α) · correlation(S, importance))
```

Higher correlation between significance and true importance yields better performance.

## Implementation Considerations

### Parameter Selection

1. **Target Rate (α)**: Based on energy budget and accuracy requirements
   - Lower α → More energy savings, potentially lower recall
   - Higher α → Better coverage, less energy savings

2. **Controller Gains**:
   - Kₚ too high → Oscillation
   - Kₚ too low → Slow convergence
   - Rule of thumb: Kᵢ ≈ Kₚ/10

3. **Hysteresis Gap**:
   - Too small → Doesn't prevent oscillation
   - Too large → Delayed response to changes
   - Typical: 2-5% of threshold range

### Computational Complexity

Per-sample complexity:
- Significance: O(k) for k features
- Gating: O(1)
- Threshold update: O(1)
- **Total: O(k)** - linear in feature count

Memory requirements:
- Activation history: O(w) for window size w
- Controller state: O(1)
- **Total: O(w)** - typically w=50-100

## Limitations and Assumptions

1. **Significance Quality**: Performance directly depends on how well S(x) correlates with true importance

2. **Distribution Shifts**: Sudden changes in input distribution can cause temporary poor performance

3. **Latency**: Not suitable for applications requiring immediate processing of all inputs

4. **Learning Period**: Needs initial samples to calibrate threshold

## Extensions

### Probabilistic Gating

Instead of hard threshold, use sigmoid:

```
P(activate) = 1 / (1 + exp(-(S-T)/τ))
```

Where τ is temperature controlling randomness.

### Multi-Objective Optimization

Extend to balance multiple goals:

```
T(t+1) = T(t) + Σᵢ λᵢ · ΔTᵢ
```

Where each ΔTᵢ optimizes different objective (energy, accuracy, latency).

### Predictive Control

Use forecast of future inputs:

```
T(t+1) = argmin E[loss(T, X_future)]
```

Requires predictive model of input stream.
