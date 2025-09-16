#!/usr/bin/env python3
"""
Test script for Sundew Auto-Tuner validation.
Compares original vs auto-tuned configurations.
"""

from src.sundew import SundewAlgorithm, get_preset


def test_auto_tuner_preset():
    """Test auto-tuned preset against baseline."""
    print("SUNDEW AUTO-TUNER VALIDATION")
    print("=" * 50)

    # Load configurations
    baseline_cfg = get_preset("tuned_v2")  # Original default
    auto_tuned_cfg = get_preset("auto_tuned")  # Auto-tuned

    print("CONFIGURATION COMPARISON")
    print("-" * 30)
    print(f"{'Parameter':<20} {'Baseline':<12} {'Auto-Tuned':<12} {'Change':<10}")
    print("-" * 30)

    key_params = [
        ('max_threshold', 'max_threshold'),
        ('ema_alpha', 'ema_alpha'),
        ('hysteresis_gap', 'hysteresis_gap'),
        ('gate_temperature', 'gate_temperature'),
        ('target_activation_rate', 'target_rate'),
        ('adapt_kp', 'kp'),
        ('adapt_ki', 'ki'),
        ('integral_clamp', 'int_clamp'),
    ]

    for param, display in key_params:
        baseline_val = getattr(baseline_cfg, param)
        tuned_val = getattr(auto_tuned_cfg, param)

        if baseline_val != 0:
            change_pct = ((tuned_val - baseline_val) / baseline_val) * 100
            change_str = f"{change_pct:+.0f}%"
        else:
            change_str = "new"

        print(f"{display:<20} {baseline_val:<12.3f} {tuned_val:<12.3f} {change_str:<10}")

    print("\nRUNNING COMPARATIVE TESTS")
    print("-" * 30)

    # Test both configurations
    configs = [
        ("Baseline", baseline_cfg),
        ("Auto-Tuned", auto_tuned_cfg)
    ]

    results = {}

    for name, cfg in configs:
        print(f"\nTesting {name} Configuration...")

        algo = SundewAlgorithm(cfg)

        # Process test events
        activations = 0
        for i in range(50):
            # Generate diverse test events
            test_event = {
                'magnitude': 30 + i * 1.5,
                'anomaly_score': 0.2 + i * 0.01,
                'context_relevance': 0.4,
                'urgency': 0.1 + i * 0.005
            }

            result = algo.process(test_event)
            if result:
                activations += 1

        # Get final report
        report = algo.report()

        results[name] = {
            'activations': activations,
            'activation_rate': activations / 50.0,
            'ema_rate': report['ema_activation_rate'],
            'threshold_final': report['threshold'],
            'threshold_util': report['threshold_utilization'],
            'energy_remaining': report['energy_remaining'],
            'energy_savings': report['estimated_energy_savings_pct'],
            'convergence_stable': report['threshold_volatility'] < 0.1
        }

        print(f"  Activations: {activations}/50 ({activations/50:.1%})")
        print(f"  EMA Rate: {report['ema_activation_rate']:.3f}")
        print(
            f"  Threshold: {report['threshold']:.3f} ({report['threshold_utilization']:.0f}% util)"
        )
        print(f"  Energy: {report['energy_remaining']:.1f}")

    print("\nCOMPARISON SUMMARY")
    print("-" * 30)

    baseline = results["Baseline"]
    tuned = results["Auto-Tuned"]

    print("TARGET METRICS:")
    print("  Activation Rate: 10-20% target")
    print(f"    Baseline: {baseline['activation_rate']:.1%}")
    print(f"    Auto-Tuned: {tuned['activation_rate']:.1%}")

    print("  Threshold Utilization: 70-85% target")
    print(f"    Baseline: {baseline['threshold_util']:.0f}%")
    print(f"    Auto-Tuned: {tuned['threshold_util']:.0f}%")

    print("  EMA vs Actual Rate Match:")
    baseline_ema_err = abs(baseline['ema_rate'] - baseline['activation_rate'])
    tuned_ema_err = abs(tuned['ema_rate'] - tuned['activation_rate'])
    print(f"    Baseline Error: {baseline_ema_err:.3f}")
    print(f"    Auto-Tuned Error: {tuned_ema_err:.3f}")

    print("  Energy Efficiency:")
    print(f"    Baseline: {baseline['energy_savings']:.1f}%")
    print(f"    Auto-Tuned: {tuned['energy_savings']:.1f}%")

    # Pass/Fail Assessment
    print("\nPASS/FAIL CHECKLIST")
    print("-" * 25)

    checks = [
        ("Activation Rate 10-20%", 0.10 <= tuned['activation_rate'] <= 0.20),
        ("Threshold Util 70-85%", 70 <= tuned['threshold_util'] <= 85),
        ("EMA Tracking Improved", tuned_ema_err < baseline_ema_err),
        ("Energy Efficiency Maintained", tuned['energy_savings'] >= baseline['energy_savings'] - 5),
        ("No Threshold Saturation", tuned['threshold_util'] < 95),
    ]

    passes = 0
    for check_name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  {check_name}: {status}")
        if passed:
            passes += 1

    print(f"\nOVERALL: {passes}/{len(checks)} checks passed")

    if passes >= 4:
        print("AUTO-TUNER SUCCESS: Ready for deployment!")
    else:
        print("AUTO-TUNER NEEDS WORK: Some targets not met")

    return results


def demo_ready_to_run_commands():
    """Show ready-to-run CLI commands."""
    print("\nREADY-TO-RUN COMMANDS")
    print("=" * 40)

    print("Basic auto-tuned demo:")
    print("  python -m sundew --demo --events 30")
    print()

    print("Conservative tuning:")
    print(
        "  python -m sundew --demo --events 30 --temperature 0.08 --ema-alpha 0.25 --hysteresis 0.03"
    )
    print()

    print("Aggressive tuning:")
    print(
        "  python -m sundew --demo --events 30 --temperature 0.12 --ema-alpha 0.40 --hysteresis 0.015"
    )
    print()

    print("Custom max threshold:")
    print("  python -m sundew --demo --events 30 --max-threshold 0.85")


if __name__ == "__main__":
    results = test_auto_tuner_preset()
    demo_ready_to_run_commands()
