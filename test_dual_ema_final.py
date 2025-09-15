#!/usr/bin/env python3
"""
Final test for dual rate metrics.
"""

from src.sundew import SundewAlgorithm, SundewConfig


def test_dual_ema_final():
    """Test dual rate EMA system."""
    print("Testing Dual Rate Metrics System...")

    config = SundewConfig(
        activation_threshold=0.9,  # High threshold to prevent activation
        target_activation_rate=0.1,
        gate_temperature=0.1,
        max_energy=100.0,
        rng_seed=42,
        dormancy_regen=(2.5, 3.5)  # Higher regen to maintain energy
    )

    algo = SundewAlgorithm(config)

    # Start with high energy and maintain it with dormancy
    algo.energy.value = 99.0

    # Process a few low-significance events (should be gated, but energy will stay high)
    for i in range(10):
        algo.process({
            'magnitude': 10,  # Low magnitude
            'anomaly_score': 0.1,
            'context_relevance': 0.1,
            'urgency': 0.1
        })

    report = algo.report()

    print(f"Energy Remaining: {report['energy_remaining']:.1f}")
    print(f"Energy Fraction: {report['energy_remaining']/100:.1%}")
    print(f"Using Slow EMA for Control: {report['ema_using_slow_for_control']}")
    print(f"EMA Fast: {report['ema_activation_rate']:.6f}")
    print(f"EMA Slow: {report['ema_activation_rate_slow']:.6f}")
    print(f"Activations: {report['activations']}")

    # Now test the actual switching by manually checking the condition
    print("\nManual threshold test:")
    energy_fraction = report['energy_remaining'] / 100.0
    should_use_slow = energy_fraction >= 0.95
    print(f"Energy fraction: {energy_fraction:.3f}")
    print(f"Should use slow EMA (>= 0.95): {should_use_slow}")
    print(f"Actually using slow EMA: {report['ema_using_slow_for_control']}")

    if should_use_slow == report['ema_using_slow_for_control']:
        print("PASS: EMA switching logic working correctly")
    else:
        print("FAIL: EMA switching logic not working")

if __name__ == "__main__":
    test_dual_ema_final()
