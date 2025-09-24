#!/usr/bin/env python3
"""
Test the simplified Sundew algorithm to verify it actually works
"""

import random
from src.sundew.simple_core import SimpleSundewAlgorithm
from src.sundew.config import SundewConfig


def generate_realistic_data(n_samples: int, anomaly_rate: float = 0.05):
    """Generate realistic test data with known patterns."""
    samples = []
    random.seed(42)  # Reproducible

    for i in range(n_samples):
        if random.random() < anomaly_rate:
            # Anomalous sample - high significance
            sample = {
                "magnitude": random.uniform(70, 100),
                "anomaly_score": random.uniform(0.7, 1.0),
                "urgency": random.uniform(0.6, 1.0),
                "context": random.uniform(0.5, 0.8)
            }
        else:
            # Normal sample - low significance
            sample = {
                "magnitude": random.uniform(10, 50),
                "anomaly_score": random.uniform(0.0, 0.3),
                "urgency": random.uniform(0.0, 0.3),
                "context": random.uniform(0.0, 0.4)
            }
        samples.append(sample)

    return samples


def test_convergence_to_targets():
    """Test if algorithm converges to different target rates."""

    print("TESTING CONVERGENCE TO MULTIPLE TARGET RATES")
    print("=" * 60)

    targets = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
    results = []

    for target in targets:
        print(f"\nTesting target rate: {target*100:.0f}%")
        print("-" * 40)

        # Create config with this target
        config = SundewConfig()
        config.target_activation_rate = target
        algorithm = SimpleSundewAlgorithm(config)

        # Generate test data
        data = generate_realistic_data(2000, anomaly_rate=0.05)

        # Process all samples
        for i, sample in enumerate(data):
            result = algorithm.process(sample)

            # Progress updates
            if (i + 1) % 400 == 0:
                report = algorithm.report()
                print(f"  Step {i+1:4d}: Rate={report['activation_rate']:.3f}, "
                      f"Threshold={report['threshold']:.3f}")

        # Final results
        final_report = algorithm.report()
        achieved_rate = final_report['activation_rate']
        error = abs(achieved_rate - target)
        success = error < 0.03  # Within 3% is good

        results.append({
            'target': target,
            'achieved': achieved_rate,
            'error': error,
            'success': success,
            'threshold': final_report['threshold'],
            'energy_saved': final_report['energy_savings_pct']
        })

        status = "PASS" if success else "FAIL"
        print(f"  FINAL: Target={target:.1%}, Achieved={achieved_rate:.1%}, "
              f"Error={error:.1%} [{status}]")
        print(f"  Energy saved: {final_report['energy_savings_pct']:.1f}%")

    # Summary
    print("\n" + "=" * 60)
    print("CONVERGENCE TEST SUMMARY")
    print("=" * 60)

    successes = sum(1 for r in results if r['success'])
    print(f"Successful convergences: {successes}/{len(results)}")

    print("\nDetailed Results:")
    print("Target  Achieved  Error    Energy_Saved  Status")
    print("-" * 50)
    for r in results:
        status = "PASS" if r['success'] else "FAIL"
        print(f"{r['target']:5.1%}   {r['achieved']:7.1%}   {r['error']:5.1%}    "
              f"{r['energy_saved']:8.1f}%     {status}")

    overall_success = successes == len(results)

    if overall_success:
        print(f"\nSUCCESS: Algorithm converges to all target rates!")
        print("The simplified algorithm is working correctly.")
    else:
        print(f"\nFAILURE: {len(results) - successes} targets failed to converge")
        print("Algorithm needs further tuning.")

    return overall_success


def test_energy_savings_demo():
    """Demonstrate energy savings with anomaly detection."""

    print("\n" + "=" * 60)
    print("ENERGY SAVINGS DEMONSTRATION")
    print("=" * 60)

    config = SundewConfig()
    config.target_activation_rate = 0.2  # Process 20% of inputs
    algorithm = SimpleSundewAlgorithm(config)

    # Generate data with 5% anomalies
    n_samples = 1000
    data = generate_realistic_data(n_samples, anomaly_rate=0.05)

    # Track anomaly detection
    anomalies_detected = 0
    total_anomalies = 0

    print(f"Processing {n_samples} samples with 5% anomalies...")
    print(f"Target: Process {config.target_activation_rate*100:.0f}% of inputs\n")

    for i, sample in enumerate(data):
        # Check if this is an anomaly (high significance)
        significance = algorithm._compute_significance(sample)
        is_anomaly = significance > 0.6  # Threshold for calling it an anomaly
        if is_anomaly:
            total_anomalies += 1

        # Process through algorithm
        result = algorithm.process(sample)

        if result is not None and is_anomaly:
            anomalies_detected += 1

        # Progress updates
        if (i + 1) % 200 == 0:
            report = algorithm.report()
            print(f"Sample {i+1:4d}: Rate={report['activation_rate']:.1%}, "
                  f"Energy_saved={report['energy_savings_pct']:.1f}%, "
                  f"Threshold={report['threshold']:.3f}")

    # Final metrics
    final_report = algorithm.report()
    anomaly_recall = anomalies_detected / total_anomalies if total_anomalies > 0 else 0

    print(f"\nFINAL RESULTS:")
    print("-" * 30)
    print(f"Total samples processed: {final_report['samples_processed']}")
    print(f"Samples activated: {final_report['samples_activated']}")
    print(f"Activation rate: {final_report['activation_rate']:.1%}")
    print(f"Energy saved: {final_report['energy_savings_pct']:.1f}%")
    print(f"Final threshold: {final_report['threshold']:.3f}")
    print(f"\nAnomaly Detection Performance:")
    print(f"Total anomalies in stream: {total_anomalies}")
    print(f"Anomalies detected: {anomalies_detected}")
    print(f"Anomaly recall: {anomaly_recall:.1%}")

    # Success criteria
    rate_ok = abs(final_report['activation_rate'] - config.target_activation_rate) < 0.05
    energy_ok = final_report['energy_savings_pct'] > 70
    anomaly_ok = anomaly_recall > 0.8

    if rate_ok and energy_ok and anomaly_ok:
        print(f"\nSUCCESS: Achieved target rate, >70% energy savings, >80% anomaly recall!")
        return True
    else:
        print(f"\nNeeds improvement:")
        if not rate_ok:
            print(f"  - Rate error too high")
        if not energy_ok:
            print(f"  - Energy savings too low")
        if not anomaly_ok:
            print(f"  - Anomaly recall too low")
        return False


if __name__ == "__main__":
    print("SIMPLIFIED SUNDEW ALGORITHM VALIDATION")
    print("=" * 60)
    print("Testing if the algorithm actually works as claimed\n")

    # Test 1: Convergence
    convergence_success = test_convergence_to_targets()

    # Test 2: Energy savings demo
    demo_success = test_energy_savings_demo()

    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)

    if convergence_success and demo_success:
        print("SUCCESS: Simplified algorithm works correctly!")
        print("- Converges to target rates reliably")
        print("- Achieves significant energy savings")
        print("- Maintains good anomaly detection performance")
        print("\nThe algorithm is ready for real-world use.")
    else:
        print("FAILURE: Algorithm still has issues:")
        if not convergence_success:
            print("- Does not converge to target rates")
        if not demo_success:
            print("- Poor energy/detection performance")
        print("\nMore tuning needed.")
