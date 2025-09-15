#!/usr/bin/env python3
"""
Simple test for dual rate metrics.
"""

from src.sundew import SundewAlgorithm, SundewConfig


def test_dual_ema_simple():
    """Test dual rate EMA system."""
    print("Testing Dual Rate Metrics System...")

    config = SundewConfig(
        activation_threshold=0.5,
        target_activation_rate=0.2,
        gate_temperature=0.1,
        max_energy=100.0,
        rng_seed=42
    )

    algo = SundewAlgorithm(config)

    # Force energy to be at capacity to test the switching logic
    algo.energy.value = 99.0  # Start at 99% to stay above 95% after consumption

    # Process one event to trigger the report calculation
    algo.process({
        'magnitude': 50,
        'anomaly_score': 0.5,
        'context_relevance': 0.4,
        'urgency': 0.3
    })

    report = algo.report()

    print(f"Energy Remaining: {report['energy_remaining']:.1f}")
    print(f"Energy Fraction: {report['energy_remaining']/100:.1%}")
    print(f"Using Slow EMA for Control: {report['ema_using_slow_for_control']}")
    print(f"EMA Fast: {report['ema_activation_rate']:.3f}")
    print(f"EMA Slow: {report['ema_activation_rate_slow']:.3f}")

    # Check if the threshold logic works correctly
    energy_fraction = report['energy_remaining'] / 100.0
    expected_slow_ema = energy_fraction >= 0.95

    print(f"Expected to use slow EMA: {expected_slow_ema}")
    print(f"Actually using slow EMA: {report['ema_using_slow_for_control']}")

    if expected_slow_ema == report['ema_using_slow_for_control']:
        print("PASS: EMA switching logic working correctly")
    else:
        print("FAIL: EMA switching logic not working")

if __name__ == "__main__":
    test_dual_ema_simple()
