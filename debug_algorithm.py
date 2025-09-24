#!/usr/bin/env python3
"""
Debug the Sundew algorithm to see why it's not reaching target rates
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from sundew.core import SundewAlgorithm
    from sundew.config import SundewConfig
    HAVE_SUNDEW = True
    print("Using real Sundew implementation")
except ImportError:
    HAVE_SUNDEW = False
    print("Using simplified implementation for debugging")

def debug_simple_case():
    """Debug with a very simple case"""

    if HAVE_SUNDEW:
        config = SundewConfig(target_activation_rate=0.2)
        algo = SundewAlgorithm(config)
    else:
        # Simple debug implementation
        class DebugAlgo:
            def __init__(self, target_rate):
                self.target_rate = target_rate
                self.threshold = 0.5
                self.history = []
                self.error_sum = 0

            def process(self, sample):
                # Compute significance
                sig = sample.get('magnitude', 0) * 0.4
                sig += sample.get('anomaly_score', 0) * 0.3
                sig += sample.get('context_relevance', 0) * 0.2
                sig += sample.get('urgency', 0) * 0.1

                print(f"  Sample: mag={sample.get('magnitude', 0):.2f}, "
                      f"sig={sig:.3f}, thresh={self.threshold:.3f}")

                # Decision
                activate = sig > self.threshold
                self.history.append(activate)

                # Update threshold every 10 samples
                if len(self.history) >= 10 and len(self.history) % 10 == 0:
                    rate = sum(self.history[-10:]) / 10
                    error = self.target_rate - rate
                    self.error_sum += error
                    old_thresh = self.threshold
                    self.threshold += 0.05 * error + 0.01 * self.error_sum  # Stronger gains
                    self.threshold = max(0.05, min(0.95, self.threshold))

                    print(f"  Threshold update: rate={rate:.3f}, error={error:.3f}, "
                          f"thresh: {old_thresh:.3f} -> {self.threshold:.3f}")

                return activate

            def report(self):
                if not self.history:
                    return {'estimated_energy_savings_pct': 0}
                rate = sum(self.history) / len(self.history)
                return {'estimated_energy_savings_pct': (1 - rate) * 100}

        algo = DebugAlgo(0.2)

    print(f"Testing with target rate: 20%")
    print("="*50)

    # Generate very simple test data
    np.random.seed(42)
    samples = []

    for i in range(100):
        if i % 10 == 0:  # 10% high importance
            sample = {
                'magnitude': 0.8,
                'anomaly_score': 0.9,
                'context_relevance': 0.7,
                'urgency': 0.8
            }
        else:  # 90% low importance
            sample = {
                'magnitude': 0.2,
                'anomaly_score': 0.1,
                'context_relevance': 0.3,
                'urgency': 0.2
            }
        samples.append(sample)

    # Process samples
    activations = 0
    for i, sample in enumerate(samples):
        if HAVE_SUNDEW:
            result = algo.process(sample)
            activated = result is not None
        else:
            activated = algo.process(sample)

        if activated:
            activations += 1

        if i < 30 or i % 20 == 0:  # Print first 30 and every 20th
            sig = sample['magnitude'] * 0.4 + sample['anomaly_score'] * 0.3 + \
                  sample['context_relevance'] * 0.2 + sample['urgency'] * 0.1
            print(f"Sample {i:2d}: sig={sig:.3f}, activated={activated}")

    final_rate = activations / len(samples)
    report = algo.report()

    print(f"\nFinal Results:")
    print(f"  Target rate: 20.0%")
    print(f"  Actual rate: {final_rate:.1%}")
    print(f"  Activations: {activations}/{len(samples)}")
    print(f"  Energy saved: {report['estimated_energy_savings_pct']:.1f}%")

    if hasattr(algo, 'threshold'):
        print(f"  Final threshold: {algo.threshold:.3f}")

def test_with_synthetic_data():
    """Test with synthetic data that should definitely work"""
    print("\nTesting with synthetic data...")
    print("="*50)

    # Very clear synthetic data
    samples = []

    # 80 low-importance samples
    for i in range(80):
        samples.append({
            'magnitude': 0.1,
            'anomaly_score': 0.05,
            'context_relevance': 0.2,
            'urgency': 0.1
        })

    # 20 high-importance samples
    for i in range(20):
        samples.append({
            'magnitude': 0.9,
            'anomaly_score': 0.95,
            'context_relevance': 0.8,
            'urgency': 0.9
        })

    # Shuffle
    np.random.shuffle(samples)

    # Test the simple algorithm
    class SimpleAlgo:
        def __init__(self, target_rate):
            self.target_rate = target_rate
            self.threshold = 0.5
            self.history = []

        def process(self, sample):
            sig = (sample['magnitude'] * 0.4 +
                   sample['anomaly_score'] * 0.3 +
                   sample['context_relevance'] * 0.2 +
                   sample['urgency'] * 0.1)

            activate = sig > self.threshold
            self.history.append(activate)

            # Update threshold
            if len(self.history) >= 20:
                rate = sum(self.history[-20:]) / 20
                error = self.target_rate - rate
                self.threshold += 0.02 * error  # Proportional only
                self.threshold = max(0.1, min(0.9, self.threshold))

            return activate

    algo = SimpleAlgo(0.2)
    activations = 0

    for sample in samples:
        if algo.process(sample):
            activations += 1

    print(f"Simple algorithm results:")
    print(f"  Activations: {activations}/100")
    print(f"  Rate: {activations/100:.1%}")
    print(f"  Final threshold: {algo.threshold:.3f}")

if __name__ == "__main__":
    debug_simple_case()
    test_with_synthetic_data()
