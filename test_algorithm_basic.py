#!/usr/bin/env python3
"""Test Sundew algorithm without visualization dependencies"""

from sundew import SundewAlgorithm, SundewConfig
import random

def test_basic_functionality():
    """Test that the algorithm actually works"""

    print("Testing Sundew Algorithm Basic Functionality")
    print("=" * 50)

    # Create algorithm with default (now fixed) configuration
    config = SundewConfig()  # Uses the fixed defaults
    algorithm = SundewAlgorithm(config)

    # Track results
    activations = []
    significances = []
    thresholds = []

    # Generate test data
    n_samples = 1000
    for i in range(n_samples):
        # Create varying patterns
        if i % 100 < 10:  # 10% high importance
            features = {
                "magnitude": random.uniform(70, 100),
                "anomaly_score": random.uniform(0.7, 1.0),
                "urgency": random.uniform(0.8, 1.0)
            }
        else:  # 90% normal data
            features = {
                "magnitude": random.uniform(10, 40),
                "anomaly_score": random.uniform(0.0, 0.3),
                "urgency": random.uniform(0.0, 0.2)
            }

        # Process through algorithm
        result = algorithm.process(features)

        # Result is ProcessingResult if activated, None if dormant
        activated = result is not None
        activations.append(activated)

        # Get significance from result or compute it
        if result:
            significances.append(result.significance)
        else:
            # Estimate significance for tracking (not exact but close)
            sig = (features.get('magnitude', 0) / 100 * 0.3 +
                   features.get('anomaly_score', 0) * 0.4 +
                   features.get('urgency', 0) * 0.1)
            significances.append(sig)

        # Track threshold from report
        report = algorithm.report()
        thresholds.append(report['threshold'])

        # Print updates
        if (i + 1) % 200 == 0:
            recent_rate = sum(activations[-100:]) / 100
            print(f"Sample {i+1:4d}: Rate={recent_rate:.2f}, Threshold={report['threshold']:.3f}")

    # Get final report
    final_report = algorithm.report()

    # Calculate final statistics
    total_activated = sum(activations)
    activation_rate = total_activated / n_samples
    energy_saved = 1.0 - activation_rate

    print("\nFinal Results:")
    print("-" * 50)
    print(f"Total samples: {n_samples}")
    print(f"Activated: {total_activated} ({activation_rate*100:.1f}%)")
    print(f"Energy saved (simple calc): {energy_saved*100:.1f}%")
    print(f"Energy saved (from report): {final_report.get('energy_savings_pct', 0):.1f}%")
    print(f"Target rate: {config.target_activation_rate*100:.1f}%")
    print(f"Final threshold: {thresholds[-1]:.3f}")
    print(f"Activation rate (from report): {final_report.get('activation_rate', 0)*100:.1f}%")

    # Check if algorithm is working correctly
    success = True
    issues = []

    # Check if it's actually gating (not all on or all off)
    if activation_rate < 0.05:
        issues.append("Algorithm is blocking almost everything (< 5% activation)")
        success = False
    elif activation_rate > 0.95:
        issues.append("Algorithm is passing almost everything (> 95% activation)")
        success = False

    # Check if it's adapting (threshold should change)
    threshold_variation = max(thresholds) - min(thresholds)
    if threshold_variation < 0.01:
        issues.append("Threshold is not adapting (variation < 0.01)")
        success = False

    # Check if it's converging to target (within 10% of target)
    if abs(activation_rate - config.target_activation_rate) > 0.1:
        issues.append(f"Failed to converge to target rate (off by {abs(activation_rate - config.target_activation_rate)*100:.1f}%)")
        success = False

    print("\nValidation:")
    print("-" * 50)
    if success:
        print("✓ Algorithm is working correctly")
        print(f"✓ Achieved {activation_rate*100:.1f}% activation (target: {config.target_activation_rate*100:.1f}%)")
        print(f"✓ Saved {energy_saved*100:.1f}% energy")
        print(f"✓ Threshold adapted from {thresholds[0]:.3f} to {thresholds[-1]:.3f}")
    else:
        print("✗ Issues detected:")
        for issue in issues:
            print(f"  - {issue}")

    return success

if __name__ == "__main__":
    success = test_basic_functionality()
    exit(0 if success else 1)
