#!/usr/bin/env python3
"""
Test enhanced telemetry and debugging features.
"""


from src.sundew import SundewAlgorithm, SundewConfig


def test_enhanced_telemetry():
    """Test enhanced telemetry collection."""
    print("Testing Enhanced Telemetry System...")

    config = SundewConfig(
        activation_threshold=0.5,
        target_activation_rate=0.15,
        gate_temperature=0.1,
        max_energy=100.0,
        rng_seed=42
    )

    algo = SundewAlgorithm(config)

    # Process events to build up telemetry data
    for i in range(30):
        _ = algo.process({
            'magnitude': 30 + i * 2,
            'anomaly_score': 0.3 + i * 0.01,
            'context_relevance': 0.4,
            'urgency': 0.2 + i * 0.005
        })

    report = algo.report()

    print(f"Total Events Processed: {report['total_inputs']}")
    print(f"Telemetry Samples: {report['telemetry_samples']}")

    print("\n=== SIGNIFICANCE STATISTICS ===")
    sig_stats = report['significance_stats']
    print(f"  Mean: {sig_stats['mean']:.3f}")
    print(f"  Std: {sig_stats['std']:.3f}")
    print(f"  Range: {sig_stats['min']:.3f} - {sig_stats['max']:.3f}")

    print("\n=== CONTROLLER STABILITY ===")
    controller = report['controller_stability']
    print(f"  Error Mean: {controller['error_mean']:.3f}")
    print(f"  Error Std: {controller['error_std']:.3f}")
    print(f"  Oscillations: {controller['oscillations']}")
    print(f"  Threshold Volatility: {report['threshold_volatility']:.3f}")

    print("\n=== ENERGY EFFICIENCY ===")
    energy_trend = report['energy_efficiency_trend']
    print(f"  Trend: {energy_trend['trend']:.3f}")
    print(f"  Efficiency: {energy_trend['efficiency']:.1%}")

    print("\n=== GATING DECISIONS ===")
    gating = report['gating_decision_breakdown']
    print(f"  Avg Gate Probability: {gating['avg_gate_prob']:.3f}")
    print(f"  Force Probe %: {gating['force_probe_pct']:.1f}%")
    print(f"  Hysteresis Influence %: {gating['hysteresis_influence']:.1f}%")

    print("\n=== DUAL RATE METRICS ===")
    print(f"  EMA Fast: {report['ema_activation_rate']:.3f}")
    print(f"  EMA Slow: {report['ema_activation_rate_slow']:.3f}")
    print(f"  Using Slow for Control: {report['ema_using_slow_for_control']}")

    print("\n=== CAP-AWARE MANAGEMENT ===")
    print(f"  Max Cap Streak: {report['max_cap_streak']}")
    print(f"  Max Low Streak: {report['max_low_streak']}")
    print(f"  Cap Nudges Applied: {report['cap_nudges_applied']}")
    print(f"  Low Nudges Applied: {report['low_nudges_applied']}")

    # Test history access
    print("\n=== HISTORY LENGTHS ===")
    print(f"  Threshold History: {len(algo.metrics.threshold_history)}")
    print(f"  Activation History: {len(algo.metrics.activation_history)}")
    print(f"  Energy History: {len(algo.metrics.energy_history)}")
    print(f"  Significance History: {len(algo.metrics.significance_history)}")
    print(f"  Controller Error History: {len(algo.metrics.controller_error_history)}")
    print(f"  Gating Decision History: {len(algo.metrics.gating_decision_history)}")

    # Show recent gating decisions
    print("\n=== RECENT GATING DECISIONS (Last 3) ===")
    for i, decision in enumerate(algo.metrics.gating_decision_history[-3:]):
        print(f"  Decision {i+1}:")
        print(f"    Significance: {decision['significance']:.3f}")
        print(f"    Threshold: {decision['threshold']:.3f}")
        print(f"    Gate Prob: {decision['gate_probability']:.3f}")
        print(f"    Activated: {decision['activated']}")
        print(f"    Energy %: {decision['energy_fraction']:.1%}")

    return report

if __name__ == "__main__":
    test_enhanced_telemetry()
    print("\nEnhanced telemetry testing complete!")
