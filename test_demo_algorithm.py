#!/usr/bin/env python3
"""Test just the core algorithm from the demo without Gradio dependencies"""

import random
from typing import List, Dict

class SundewDemo:
    """Simplified Sundew algorithm for demonstration"""

    def __init__(self, target_rate: float = 0.2):
        self.target_rate = target_rate
        self.threshold = 0.5
        self.activation_history = []
        self.error_sum = 0
        self.hysteresis_gap = 0.02
        self.was_active = False

    def compute_significance(self, sample: Dict[str, float]) -> float:
        """Compute significance score (0-1) from sample features"""
        sig = 0.4 * (sample['magnitude'] / 100)
        sig += 0.3 * sample['anomaly']
        sig += 0.2 * sample['urgency']
        sig += 0.1 * sample['trend']
        return min(1.0, max(0.0, sig))

    def process_sample(self, sample: Dict[str, float]) -> bool:
        """Process one sample and return activation decision"""

        # Compute significance
        significance = self.compute_significance(sample)

        # Apply hysteresis to threshold
        if self.was_active:
            effective_threshold = self.threshold - self.hysteresis_gap
        else:
            effective_threshold = self.threshold + self.hysteresis_gap

        # Make activation decision
        activate = significance > effective_threshold

        # Update state
        self.activation_history.append(activate)
        self.was_active = activate

        # Update threshold (PI controller) - FIXED: Correct direction and stronger gains
        if len(self.activation_history) >= 10:
            recent_rate = sum(self.activation_history[-10:]) / 10
            error = self.target_rate - recent_rate
            self.error_sum += error
            # Prevent integral windup
            self.error_sum = max(-5.0, min(5.0, self.error_sum))

            # PI update - FIXED: SUBTRACT error to decrease threshold when rate is too low
            adjustment = 0.05 * error + 0.002 * self.error_sum
            self.threshold -= adjustment  # FIXED: Changed += to -=
            self.threshold = min(0.95, max(0.05, self.threshold))

        return activate

def generate_sample_stream(n_samples: int, anomaly_rate: float = 0.1) -> List[Dict[str, float]]:
    """Generate synthetic data stream with known patterns"""
    samples = []
    random.seed(42)

    for i in range(n_samples):
        # Create occasional high-importance events
        if i % int(1/anomaly_rate) == 0:  # Anomaly
            sample = {
                'magnitude': random.uniform(70, 100),
                'anomaly': random.uniform(0.7, 1.0),
                'urgency': random.uniform(0.8, 1.0),
                'trend': random.uniform(0.6, 0.9)
            }
        else:  # Normal sample
            sample = {
                'magnitude': random.uniform(10, 40),
                'anomaly': random.uniform(0.0, 0.3),
                'urgency': random.uniform(0.0, 0.2),
                'trend': random.uniform(0.2, 0.5)
            }

        samples.append(sample)

    return samples

def test_demo_algorithm():
    """Test the demo algorithm to see if it converges"""

    print("Testing Sundew Demo Algorithm")
    print("=" * 50)

    # Test different target rates
    targets = [0.1, 0.2, 0.3]

    for target in targets:
        print(f"\nTarget Rate: {target*100:.0f}%")
        print("-" * 30)

        algo = SundewDemo(target_rate=target)
        samples = generate_sample_stream(1000, 0.1)

        activations = 0
        for i, sample in enumerate(samples):
            if algo.process_sample(sample):
                activations += 1

            # Progress updates
            if (i + 1) % 200 == 0:
                rate = activations / (i + 1)
                print(f"Step {i+1:4d}: Rate={rate:.1%}, Threshold={algo.threshold:.3f}")

        final_rate = activations / len(samples)
        error = abs(final_rate - target)
        success = error < 0.05

        print(f"Final: Target={target:.1%}, Achieved={final_rate:.1%}, Error={error:.1%}")
        print(f"Status: {'OK' if success else 'FAIL'}")

if __name__ == "__main__":
    test_demo_algorithm()
