#!/usr/bin/env python3
"""Quick test of 40% target rate with wider bounds"""

import random
from src.sundew.simple_core import SimpleSundewAlgorithm
from src.sundew.config import SundewConfig


def generate_realistic_data(n_samples: int, anomaly_rate: float = 0.05):
    """Generate realistic test data with known patterns."""
    samples = []
    random.seed(42)  # Reproducible

    for i in range(n_samples):
        if random.random() < anomaly_rate:
            # Anomalous sample - high significance
            sample = {
                "magnitude": random.uniform(70, 100),
                "anomaly_score": random.uniform(0.7, 1.0),
                "urgency": random.uniform(0.6, 1.0),
                "context": random.uniform(0.5, 0.8)
            }
        else:
            # Normal sample - low significance
            sample = {
                "magnitude": random.uniform(10, 50),
                "anomaly_score": random.uniform(0.0, 0.3),
                "urgency": random.uniform(0.0, 0.3),
                "context": random.uniform(0.0, 0.4)
            }
        samples.append(sample)

    return samples


def test_40_percent():
    """Test 40% target specifically"""

    print("Testing 40% target rate with wider threshold bounds")
    print("=" * 50)

    config = SundewConfig()
    config.target_activation_rate = 0.4
    algorithm = SimpleSundewAlgorithm(config)

    print(f"Config: min_threshold={config.min_threshold}, max_threshold={config.max_threshold}")
    print(f"Starting threshold: {algorithm.threshold}")
    print()

    # Generate test data
    data = generate_realistic_data(2000, anomaly_rate=0.05)

    # Process all samples
    for i, sample in enumerate(data):
        result = algorithm.process(sample)

        # Progress updates
        if (i + 1) % 400 == 0:
            report = algorithm.report()
            print(f"Step {i+1:4d}: Rate={report['activation_rate']:.3f}, "
                  f"Threshold={report['threshold']:.3f}")

    # Final results
    final_report = algorithm.report()
    achieved_rate = final_report['activation_rate']
    error = abs(achieved_rate - 0.4)
    success = error < 0.03

    print(f"\nFINAL RESULTS:")
    print(f"Target: 40.0%, Achieved: {achieved_rate:.1%}, Error: {error:.1%}")
    print(f"Energy saved: {final_report['energy_savings_pct']:.1f}%")
    print(f"Final threshold: {final_report['threshold']:.3f}")
    print(f"Status: {'PASS' if success else 'FAIL'}")

    return success


if __name__ == "__main__":
    test_40_percent()
