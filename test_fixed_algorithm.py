#!/usr/bin/env python3
"""
Test Sundew algorithm with fixed parameters
"""

import sys
import numpy as np
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sundew.core import SundewAlgorithm
from sundew.config import SundewConfig

def test_with_better_config():
    """Test with faster adapting configuration"""

    print("Testing Sundew with improved configuration")
    print("="*50)

    # Create config with faster adaptation
    config = SundewConfig(
        target_activation_rate=0.2,
        activation_threshold=0.5,  # Start in middle
        adapt_kp=0.05,  # Much higher proportional gain
        adapt_ki=0.01,  # Higher integral gain
        min_threshold=0.1,  # Lower bound
        max_threshold=0.9,  # Higher bound
        hysteresis_gap=0.02
    )

    algo = SundewAlgorithm(config)

    # Generate clear test data
    np.random.seed(42)
    samples = []
    labels = []

    for i in range(200):
        if i % 10 == 0:  # 10% high-importance
            sample = {
                'magnitude': 80,  # High values
                'anomaly_score': 0.9,
                'context_relevance': 0.8,
                'urgency': 0.9
            }
            labels.append(True)
        else:  # 90% low-importance
            sample = {
                'magnitude': 20,  # Low values
                'anomaly_score': 0.1,
                'context_relevance': 0.3,
                'urgency': 0.2
            }
            labels.append(False)
        samples.append(sample)

    # Process samples
    activations = []
    thresholds = []

    for i, sample in enumerate(samples):
        result = algo.process(sample)
        activated = result is not None
        activations.append(activated)

        # Get current threshold
        if hasattr(algo, 'threshold'):
            thresholds.append(algo.threshold)

        # Print some progress
        if i < 50 and i % 10 == 0:
            rate = sum(activations) / (i + 1) if i > 0 else 0
            thresh = thresholds[-1] if thresholds else "N/A"
            print(f"Sample {i:2d}: activated={activated}, rate={rate:.3f}, threshold={thresh:.3f}")

    # Calculate final metrics
    total_activations = sum(activations)
    final_rate = total_activations / len(samples)

    # Calculate precision/recall on the high-importance samples
    tp = sum(1 for a, l in zip(activations, labels) if a and l)
    fp = sum(1 for a, l in zip(activations, labels) if a and not l)
    fn = sum(1 for a, l in zip(activations, labels) if not a and l)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nFinal Results:")
    print(f"  Target rate: 20.0%")
    print(f"  Actual rate: {final_rate:.1%}")
    print(f"  Total activations: {total_activations}/{len(samples)}")
    print(f"  Energy saved: {(1-final_rate)*100:.1f}%")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1 Score: {f1:.3f}")
    if thresholds:
        print(f"  Final threshold: {thresholds[-1]:.3f}")

    # Test different target rates
    print(f"\nTesting different target rates:")
    print("-" * 30)

    for target in [0.1, 0.15, 0.2, 0.25, 0.3]:
        config_test = SundewConfig(
            target_activation_rate=target,
            activation_threshold=0.5,
            adapt_kp=0.05,
            adapt_ki=0.01,
            min_threshold=0.1,
            max_threshold=0.9
        )

        algo_test = SundewAlgorithm(config_test)

        test_activations = 0
        for sample in samples:
            if algo_test.process(sample) is not None:
                test_activations += 1

        actual_rate = test_activations / len(samples)
        energy_saved = (1 - actual_rate) * 100

        print(f"  Target {target:.0%}: Actual {actual_rate:.1%}, Energy saved {energy_saved:.1f}%")

def test_on_realistic_data():
    """Test on more realistic streaming data"""

    print(f"\nTesting on realistic streaming data")
    print("="*50)

    # Better config
    config = SundewConfig(
        target_activation_rate=0.2,
        activation_threshold=0.5,
        adapt_kp=0.03,  # Moderate gains
        adapt_ki=0.005,
        min_threshold=0.1,
        max_threshold=0.9
    )

    algo = SundewAlgorithm(config)

    # Generate more realistic data with varying patterns
    np.random.seed(42)
    samples = []
    labels = []

    for i in range(1000):
        # Create time-varying anomaly rate
        time_factor = (i / 1000)
        anomaly_prob = 0.05 + 0.1 * np.sin(time_factor * 4 * np.pi)  # 5-15% anomalies

        is_anomaly = np.random.random() < anomaly_prob

        if is_anomaly:
            # Anomalous sample
            sample = {
                'magnitude': np.random.uniform(60, 100),
                'anomaly_score': np.random.uniform(0.7, 1.0),
                'context_relevance': np.random.uniform(0.6, 0.9),
                'urgency': np.random.uniform(0.7, 1.0)
            }
        else:
            # Normal sample
            sample = {
                'magnitude': np.random.uniform(10, 40),
                'anomaly_score': np.random.uniform(0.0, 0.3),
                'context_relevance': np.random.uniform(0.2, 0.5),
                'urgency': np.random.uniform(0.1, 0.3)
            }

        samples.append(sample)
        labels.append(is_anomaly)

    # Process
    activations = []
    for sample in samples:
        result = algo.process(sample)
        activations.append(result is not None)

    # Metrics
    total_activations = sum(activations)
    final_rate = total_activations / len(samples)

    tp = sum(1 for a, l in zip(activations, labels) if a and l)
    fp = sum(1 for a, l in zip(activations, labels) if a and not l)
    fn = sum(1 for a, l in zip(activations, labels) if not a and l)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    total_anomalies = sum(labels)

    print(f"Results on realistic data:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Total anomalies: {total_anomalies} ({100*total_anomalies/len(samples):.1f}%)")
    print(f"  Target rate: 20.0%")
    print(f"  Actual rate: {final_rate:.1%}")
    print(f"  Energy saved: {(1-final_rate)*100:.1f}%")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1 Score: {f1:.3f}")

    # Check rate stability over time
    window_size = 100
    rates = []
    for i in range(window_size, len(activations), window_size):
        window_rate = sum(activations[i-window_size:i]) / window_size
        rates.append(window_rate)

    if rates:
        print(f"  Rate stability: {np.std(rates):.3f} (std dev across windows)")

if __name__ == "__main__":
    test_with_better_config()
    test_on_realistic_data()
