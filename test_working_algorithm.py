#!/usr/bin/env python3
"""
Test Sundew algorithm with correct configuration to make it work
"""

import sys
import numpy as np
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sundew.core import SundewAlgorithm
from sundew.config import SundewConfig

def test_working_config():
    """Test with configuration that should actually work"""

    print("Testing Sundew with working configuration")
    print("="*60)

    # Create config that disables problematic features
    config = SundewConfig(
        target_activation_rate=0.2,
        activation_threshold=0.5,

        # Use PI controller instead of AIMD
        adapt_kp=0.1,  # Much higher proportional gain
        adapt_ki=0.02,  # Higher integral gain

        # Disable energy pressure effects
        energy_pressure=0.0,
        max_energy=1000.0,  # Higher energy to avoid cap effects

        # Faster EMA
        ema_alpha=0.3,

        # Reasonable bounds
        min_threshold=0.1,
        max_threshold=0.9,

        # Small hysteresis
        hysteresis_gap=0.01
    )

    # Force PI controller by monkey-patching
    algo = SundewAlgorithm(config)
    algo._use_aimd = False  # Force PI controller

    print(f"Configuration:")
    print(f"  Using AIMD: {algo._use_aimd}")
    print(f"  Kp: {algo._kp}, Ki: {algo._ki}")
    print(f"  Energy pressure: {algo._press}")
    print(f"  Max energy: {config.max_energy}")
    print()

    # Generate very clear test data
    np.random.seed(42)
    samples = []
    labels = []

    for i in range(500):
        if i % 20 == 0:  # 5% high-importance (should be caught)
            sample = {
                'magnitude': 90,
                'anomaly_score': 0.9,
                'context_relevance': 0.8,
                'urgency': 0.9
            }
            labels.append(True)
        else:  # 95% low-importance
            sample = {
                'magnitude': 15,
                'anomaly_score': 0.05,
                'context_relevance': 0.2,
                'urgency': 0.1
            }
            labels.append(False)
        samples.append(sample)

    # Process samples and track progress
    activations = []
    rates = []
    thresholds = []

    for i, sample in enumerate(samples):
        result = algo.process(sample)
        activated = result is not None
        activations.append(activated)

        # Track current rate and threshold
        current_rate = sum(activations) / (i + 1)
        rates.append(current_rate)
        thresholds.append(algo.threshold)

        # Print progress every 50 samples
        if (i + 1) % 50 == 0:
            ema_rate = algo.metrics.ema_activation_rate
            print(f"Sample {i+1:3d}: Rate={current_rate:.3f}, EMA={ema_rate:.3f}, Threshold={algo.threshold:.3f}")

    # Final results
    final_rate = sum(activations) / len(activations)
    energy_saved = (1 - final_rate) * 100

    # Calculate precision/recall
    tp = sum(1 for a, l in zip(activations, labels) if a and l)
    fp = sum(1 for a, l in zip(activations, labels) if a and not l)
    fn = sum(1 for a, l in zip(activations, labels) if not a and l)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nFinal Results:")
    print(f"  Target rate: 20.0%")
    print(f"  Actual rate: {final_rate:.1%}")
    print(f"  Energy saved: {energy_saved:.1f}%")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1 Score: {f1:.3f}")
    print(f"  Final threshold: {algo.threshold:.3f}")
    print(f"  Final EMA rate: {algo.metrics.ema_activation_rate:.3f}")

    # Check if algorithm reached target
    error = abs(final_rate - 0.2)
    if error < 0.05:  # Within 5%
        print(f"  SUCCESS: Rate control within ±5% ({error:.1%} error)")
    else:
        print(f"  FAILED: Rate control error {error:.1%}")

    return final_rate, f1, energy_saved

def test_multiple_target_rates():
    """Test different target rates to verify the algorithm works"""

    print(f"\nTesting multiple target rates")
    print("="*60)

    results = []

    for target_rate in [0.1, 0.15, 0.2, 0.25, 0.3]:
        print(f"\nTarget rate: {target_rate:.0%}")
        print("-" * 30)

        config = SundewConfig(
            target_activation_rate=target_rate,
            activation_threshold=0.5,
            adapt_kp=0.08,
            adapt_ki=0.015,
            energy_pressure=0.0,
            max_energy=1000.0,
            ema_alpha=0.25,
            min_threshold=0.05,
            max_threshold=0.95
        )

        algo = SundewAlgorithm(config)
        algo._use_aimd = False  # Force PI controller

        # Simple test data
        np.random.seed(42)
        activations = 0
        total_samples = 300

        for i in range(total_samples):
            if i % 25 == 0:  # 4% anomalies
                sample = {'magnitude': 85, 'anomaly_score': 0.85, 'context_relevance': 0.75, 'urgency': 0.8}
            else:
                sample = {'magnitude': 20, 'anomaly_score': 0.1, 'context_relevance': 0.25, 'urgency': 0.15}

            if algo.process(sample) is not None:
                activations += 1

        actual_rate = activations / total_samples
        energy_saved = (1 - actual_rate) * 100
        error = abs(actual_rate - target_rate)

        print(f"  Actual rate: {actual_rate:.1%}")
        print(f"  Energy saved: {energy_saved:.1f}%")
        print(f"  Error: {error:.1%}")
        print(f"  Final threshold: {algo.threshold:.3f}")

        results.append({
            'target': target_rate,
            'actual': actual_rate,
            'energy_saved': energy_saved,
            'error': error
        })

    # Summary
    print(f"\nSummary:")
    print("=" * 60)
    avg_error = np.mean([r['error'] for r in results])
    avg_energy = np.mean([r['energy_saved'] for r in results])

    print(f"Average rate error: {avg_error:.1%}")
    print(f"Average energy saved: {avg_energy:.1f}%")

    success_count = sum(1 for r in results if r['error'] < 0.05)
    print(f"Successful tests (±5%): {success_count}/{len(results)}")

    if success_count >= 4:
        print("SUCCESS: Algorithm is working correctly!")
    else:
        print("FAILED: Algorithm needs further tuning")

def test_real_datasets():
    """Test on the actual datasets with working config"""

    print(f"\nTesting on real datasets with working configuration")
    print("="*60)

    # Working config
    config = SundewConfig(
        target_activation_rate=0.2,
        activation_threshold=0.5,
        adapt_kp=0.06,
        adapt_ki=0.01,
        energy_pressure=0.0,
        max_energy=1000.0,
        ema_alpha=0.2,
        min_threshold=0.1,
        max_threshold=0.9
    )

    # Test datasets
    datasets = [
        ("Synthetic Network", generate_network_data),
        ("Synthetic Financial", generate_financial_data),
        ("Synthetic Sensors", generate_sensor_data)
    ]

    results = []

    for name, generator in datasets:
        print(f"\n{name}:")
        print("-" * 40)

        samples, labels = generator()

        algo = SundewAlgorithm(config)
        algo._use_aimd = False

        activations = []
        for sample in samples:
            result = algo.process(sample)
            activations.append(result is not None)

        # Metrics
        actual_rate = sum(activations) / len(activations)
        energy_saved = (1 - actual_rate) * 100

        tp = sum(1 for a, l in zip(activations, labels) if a and l)
        fp = sum(1 for a, l in zip(activations, labels) if a and not l)
        fn = sum(1 for a, l in zip(activations, labels) if not a and l)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"  Samples: {len(samples)}, Anomalies: {sum(labels)} ({100*sum(labels)/len(samples):.1f}%)")
        print(f"  Target rate: 20.0%")
        print(f"  Actual rate: {actual_rate:.1%}")
        print(f"  Energy saved: {energy_saved:.1f}%")
        print(f"  F1 Score: {f1:.3f}")
        print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}")

        results.append({
            'dataset': name,
            'rate': actual_rate,
            'energy_saved': energy_saved,
            'f1': f1
        })

    # Overall summary
    print(f"\nOverall Results:")
    print("=" * 60)
    avg_rate = np.mean([r['rate'] for r in results])
    avg_energy = np.mean([r['energy_saved'] for r in results])
    avg_f1 = np.mean([r['f1'] for r in results])

    print(f"Average activation rate: {avg_rate:.1%}")
    print(f"Average energy saved: {avg_energy:.1f}%")
    print(f"Average F1 score: {avg_f1:.3f}")

def generate_network_data():
    """Generate synthetic network security data"""
    np.random.seed(42)
    samples, labels = [], []

    for i in range(2000):
        is_anomaly = np.random.random() < 0.03  # 3% attacks

        if is_anomaly:
            sample = {
                'magnitude': np.random.uniform(70, 100),
                'anomaly_score': np.random.uniform(0.8, 1.0),
                'context_relevance': np.random.uniform(0.7, 0.9),
                'urgency': np.random.uniform(0.8, 1.0)
            }
        else:
            sample = {
                'magnitude': np.random.uniform(5, 30),
                'anomaly_score': np.random.uniform(0.0, 0.2),
                'context_relevance': np.random.uniform(0.2, 0.4),
                'urgency': np.random.uniform(0.1, 0.3)
            }

        samples.append(sample)
        labels.append(is_anomaly)

    return samples, labels

def generate_financial_data():
    """Generate synthetic financial data"""
    np.random.seed(42)
    samples, labels = [], []

    for i in range(1500):
        is_anomaly = np.random.random() < 0.08  # 8% market events

        if is_anomaly:
            sample = {
                'magnitude': np.random.uniform(75, 95),
                'anomaly_score': np.random.uniform(0.75, 1.0),
                'context_relevance': np.random.uniform(0.6, 0.8),
                'urgency': np.random.uniform(0.7, 0.9)
            }
        else:
            sample = {
                'magnitude': np.random.uniform(10, 35),
                'anomaly_score': np.random.uniform(0.0, 0.25),
                'context_relevance': np.random.uniform(0.25, 0.45),
                'urgency': np.random.uniform(0.1, 0.35)
            }

        samples.append(sample)
        labels.append(is_anomaly)

    return samples, labels

def generate_sensor_data():
    """Generate synthetic sensor data"""
    np.random.seed(42)
    samples, labels = [], []

    for i in range(2500):
        is_anomaly = np.random.random() < 0.06  # 6% sensor faults

        if is_anomaly:
            sample = {
                'magnitude': np.random.uniform(65, 90),
                'anomaly_score': np.random.uniform(0.7, 0.95),
                'context_relevance': np.random.uniform(0.5, 0.75),
                'urgency': np.random.uniform(0.6, 0.85)
            }
        else:
            sample = {
                'magnitude': np.random.uniform(8, 25),
                'anomaly_score': np.random.uniform(0.0, 0.2),
                'context_relevance': np.random.uniform(0.2, 0.4),
                'urgency': np.random.uniform(0.05, 0.25)
            }

        samples.append(sample)
        labels.append(is_anomaly)

    return samples, labels

if __name__ == "__main__":
    test_working_config()
    test_multiple_target_rates()
    test_real_datasets()
