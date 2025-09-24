#!/usr/bin/env python3
"""
ACTUALLY WORKING implementation of Sundew
The key fix: stronger adaptation gains and proper integral windup prevention
"""

import random

class ActuallyWorkingSundew:
    """This version actually converges to target rates"""

    def __init__(self, target_rate=0.2):
        self.target_rate = target_rate
        self.threshold = 0.5  # Start in middle
        # MUCH stronger gains needed for convergence
        self.kp = 0.05   # Was way too weak before
        self.ki = 0.002
        self.error_sum = 0
        self.max_integral = 5.0  # Prevent windup
        self.history = []

    def process(self, features):
        # Compute significance
        sig = features['significance']

        # Simple gate decision
        activate = sig > self.threshold
        self.history.append(1 if activate else 0)

        # Adapt every 10 samples (more frequent updates)
        if len(self.history) >= 10 and len(self.history) % 10 == 0:
            # Use recent window for current rate
            window = min(50, len(self.history))
            recent = self.history[-window:]
            current_rate = sum(recent) / len(recent)

            # PI controller
            error = self.target_rate - current_rate
            self.error_sum += error
            # Prevent integral windup
            self.error_sum = max(-self.max_integral, min(self.max_integral, self.error_sum))

            # Update threshold (DECREASE threshold to activate MORE)
            adjustment = self.kp * error + self.ki * self.error_sum
            self.threshold -= adjustment

            # Keep in valid range
            self.threshold = max(0.05, min(0.95, self.threshold))

        return activate

def generate_realistic_data(n=1000):
    """Generate data with known significance distribution"""
    data = []
    random.seed(42)  # Reproducible

    for i in range(n):
        if random.random() < 0.1:  # 10% interesting
            sig = random.uniform(0.6, 1.0)
        else:  # 90% boring
            sig = random.uniform(0.0, 0.4)
        data.append({'significance': sig})

    return data

def test_convergence():
    """Test algorithm convergence to different target rates"""

    print("SUNDEW CONVERGENCE TEST")
    print("=" * 70)
    print("Testing if algorithm can actually reach target activation rates")
    print()

    target_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = []

    for target in target_rates:
        # Fresh algorithm instance
        algo = ActuallyWorkingSundew(target_rate=target)

        # Generate consistent test data
        data = generate_realistic_data(2000)

        # Process all samples
        activations = []
        for sample in data:
            activated = algo.process(sample)
            activations.append(1 if activated else 0)

        # Calculate final rate (last 500 samples for stability)
        final_rate = sum(activations[-500:]) / 500
        error = abs(final_rate - target)
        success = error < 0.05  # Within 5% is success

        results.append({
            'target': target,
            'achieved': final_rate,
            'error': error,
            'success': success,
            'threshold': algo.threshold
        })

        status = "OK" if success else "FAIL"
        print(f"Target: {target:4.1%} -> Achieved: {final_rate:4.1%} "
              f"(error: {error:4.1%}) [{status}] "
              f"Final threshold: {algo.threshold:.3f}")

    # Summary
    successes = sum(1 for r in results if r['success'])
    print()
    print("-" * 70)
    print(f"RESULTS: {successes}/{len(results)} targets achieved")

    if successes == len(results):
        print("SUCCESS: Algorithm converges properly to all target rates!")
    else:
        print("FAILURE: Algorithm does not converge reliably")

    return successes == len(results)

def demonstrate_energy_savings():
    """Show actual energy savings in action"""

    print("\nENERGY SAVINGS DEMONSTRATION")
    print("=" * 70)
    print("Processing stream with 5% anomalies, targeting 20% activation")
    print()

    algo = ActuallyWorkingSundew(target_rate=0.2)

    # Generate stream with known anomalies
    n_samples = 1000
    anomalies_detected = 0
    total_anomalies = 0
    activations = 0

    random.seed(123)  # Different seed for variety

    for i in range(n_samples):
        # 5% are actual anomalies
        is_anomaly = random.random() < 0.05
        if is_anomaly:
            total_anomalies += 1
            sig = random.uniform(0.7, 1.0)  # High significance
        else:
            sig = random.uniform(0.0, 0.5)   # Low significance

        activated = algo.process({'significance': sig})

        if activated:
            activations += 1
            if is_anomaly:
                anomalies_detected += 1

        # Progress
        if (i + 1) % 200 == 0:
            rate = activations / (i + 1)
            print(f"Samples: {i+1:4d} | Activation rate: {rate:5.1%} | "
                  f"Threshold: {algo.threshold:.3f}")

    # Final metrics
    activation_rate = activations / n_samples
    energy_saved = 1 - activation_rate
    anomaly_recall = anomalies_detected / total_anomalies if total_anomalies > 0 else 0

    print()
    print("FINAL METRICS:")
    print("-" * 70)
    print(f"Target activation rate:  {algo.target_rate:5.1%}")
    print(f"Actual activation rate:  {activation_rate:5.1%}")
    print(f"Energy saved:           {energy_saved:5.1%}")
    print(f"Anomalies in stream:    {total_anomalies}/{n_samples} ({total_anomalies/n_samples:4.1%})")
    print(f"Anomalies detected:     {anomalies_detected}/{total_anomalies} ({anomaly_recall:4.1%})")
    print()

    if activation_rate < 0.25 and anomaly_recall > 0.8:
        print("SUCCESS: Saved >75% energy while detecting >80% of anomalies!")
    else:
        print("Algorithm needs tuning for this use case")

if __name__ == "__main__":
    print("=" * 70)
    print("SUNDEW ALGORITHM - WORKING DEMONSTRATION")
    print("=" * 70)
    print()

    # Test 1: Convergence
    convergence_ok = test_convergence()

    print()

    # Test 2: Energy savings
    demonstrate_energy_savings()

    print()
    print("=" * 70)
    if convergence_ok:
        print("CONCLUSION: Algorithm is WORKING and achieves claimed performance")
        print("The key was using appropriate control gains for convergence")
    else:
        print("CONCLUSION: Algorithm needs parameter tuning")
