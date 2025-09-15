#!/usr/bin/env python3
"""
Test script to verify energy accounting bug fixes.
"""

from src.sundew import SundewAlgorithm, SundewConfig


def test_energy_accounting():
    """Test that energy accounting is consistent."""
    print("Testing Energy Accounting Fix...")

    config = SundewConfig(
        activation_threshold=0.5,
        target_activation_rate=0.2,
        gate_temperature=0.1,
        max_energy=100.0,
        rng_seed=42
    )

    algo = SundewAlgorithm(config)

    # Process some events
    for i in range(20):
        result = algo.process({
            'magnitude': 30 + i * 2,
            'anomaly_score': 0.3 + i * 0.02,
            'context_relevance': 0.4,
            'urgency': 0.2
        })
        if result:
            print(f"  Event {i+1}: ACTIVATED (sig={result.significance:.3f})")
        else:
            print(f"  Event {i+1}: gated")

    # Get final report
    report = algo.report()

    print("\nEnergy Accounting Verification:")
    print(f"  Total Energy Spent: {report['total_energy_spent']:.1f}")
    print(f"  Energy Recovered: {report['energy_recovered']:.1f}")
    print(f"  Net Energy Consumed: {report['net_energy_consumed']:.1f}")
    print(f"  Energy Remaining: {report['energy_remaining']:.1f}")
    print(f"  Processing Energy: {report['energy_spent_processing']:.1f}")
    print(f"  Dormancy Energy: {report['energy_spent_dormancy']:.1f}")

    # Verify consistency
    expected_remaining = 100.0 - report['net_energy_consumed']
    actual_remaining = report['energy_remaining']

    print("\nConsistency Check:")
    print(f"  Expected Remaining: {expected_remaining:.1f}")
    print(f"  Actual Remaining: {actual_remaining:.1f}")
    print(f"  Difference: {abs(expected_remaining - actual_remaining):.3f}")

    # Check that total spent is reasonable (should be > 0 now)
    if report['total_energy_spent'] > 0:
        print(f"  PASS: Total energy spent is > 0: {report['total_energy_spent']:.1f}")
    else:
        print("  FAIL: Total energy spent is still 0 - bug not fixed!")

    print("\nController Status:")
    print(f"  EMA Alpha: {report['ema_alpha']:.3f}")
    print(f"  EMA Discrepancy: {report['ema_discrepancy']:.3f}")
    print(f"  Threshold Utilization: {report['threshold_utilization']:.1%}")
    print(f"  Hysteresis Gap: {report['hysteresis_gap']:.3f}")

    return report

def test_hysteresis():
    """Test hysteresis behavior."""
    print("\nTesting Hysteresis Fix...")

    from src.sundew.gating import gate_probability_with_hysteresis

    # Test hysteresis behavior
    sig = 0.75
    threshold = 0.7
    temp = 0.1
    gap = 0.05

    # When not activated, harder to activate
    p_inactive = gate_probability_with_hysteresis(sig, threshold, temp, False, gap)

    # When already activated, easier to stay activated
    p_active = gate_probability_with_hysteresis(sig, threshold, temp, True, gap)

    print(f"  Significance: {sig}")
    print(f"  Threshold: {threshold}")
    print(f"  P(activate | inactive): {p_inactive:.3f}")
    print(f"  P(activate | active): {p_active:.3f}")
    print(f"  Hysteresis effect: {p_active - p_inactive:.3f}")

    if p_active > p_inactive:
        print("  PASS: Hysteresis working - easier to stay activated")
    else:
        print("  FAIL: Hysteresis not working properly")

if __name__ == "__main__":
    test_energy_accounting()
    test_hysteresis()
    print("\nBug fix testing complete!")
