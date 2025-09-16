#!/usr/bin/env python3
"""
Test script to verify dual rate metrics implementation.
"""

from src.sundew import SundewAlgorithm, SundewConfig


def test_dual_ema():
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

    # Process some events to build up EMA values
    for i in range(50):
        result = algo.process({
            'magnitude': 30 + i,
            'anomaly_score': 0.3 + i * 0.01,
            'context_relevance': 0.4,
            'urgency': 0.2
        })

    # Get final report
    report = algo.report()

    print("\nDual Rate Metrics Results:")
    print(f"  Activation Rate (actual): {report['activation_rate']:.1%}")
    print(f"  EMA Fast (a=0.15): {report['ema_activation_rate']:.3f}")
    print(f"  EMA Slow (a=0.05): {report['ema_activation_rate_slow']:.3f}")
    print(f"  Using Slow EMA for Control: {report['ema_using_slow_for_control']}")
    print(f"  Energy Remaining: {report['energy_remaining']:.1f}")
    print(f"  Energy Fraction: {report['energy_remaining']/100:.1%}")

    # Test near-capacity behavior
    print("\nTesting Near-Capacity Behavior...")

    # Set energy to near capacity to test slow EMA usage
    algo.energy.value = 96.0  # 96% capacity

    # Process a few more events
    for i in range(5):
        result = algo.process({
            'magnitude': 50,
            'anomaly_score': 0.5,
            'context_relevance': 0.4,
            'urgency': 0.3
        })

    report_near_cap = algo.report()

    print(f"  Energy After Near-Cap Test: {report_near_cap['energy_remaining']:.1f}")
    print(f"  Energy Fraction: {report_near_cap['energy_remaining']/100:.1%}")
    print(f"  Using Slow EMA for Control: {report_near_cap['ema_using_slow_for_control']}")
    print(f"  EMA Fast: {report_near_cap['ema_activation_rate']:.3f}")
    print(f"  EMA Slow: {report_near_cap['ema_activation_rate_slow']:.3f}")

    if report_near_cap['ema_using_slow_for_control']:
        print("  PASS: Using slow EMA when energy near capacity")
    else:
        print("  FAIL: Should use slow EMA when energy near capacity")

    # Test normal capacity behavior
    algo.energy.value = 80.0  # 80% capacity - normal range

    for i in range(3):
        result = algo.process({
            'magnitude': 40,
            'anomaly_score': 0.4,
            'context_relevance': 0.3,
            'urgency': 0.2
        })

    report_normal = algo.report()

    print("\nNormal Capacity Test:")
    print(f"  Energy: {report_normal['energy_remaining']:.1f}")
    print(f"  Energy Fraction: {report_normal['energy_remaining']/100:.1%}")
    print(f"  Using Slow EMA for Control: {report_normal['ema_using_slow_for_control']}")

    if not report_normal['ema_using_slow_for_control']:
        print("  PASS: Using fast EMA when energy in normal range")
    else:
        print("  FAIL: Should use fast EMA when energy in normal range")

    return report_normal

if __name__ == "__main__":
    test_dual_ema()
    print("\nDual rate metrics testing complete!")
