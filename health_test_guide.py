#!/usr/bin/env python3
"""
[HEALTH] COMPREHENSIVE HEALTH MONITORING TEST GUIDE
==============================================

Complete testing guide for life-saving maternal health monitoring
and cardiac monitoring systems using Sundew algorithms.

Usage:
    python health_test_guide.py --test [maternal|cardiac|all]
    python health_test_guide.py --demo
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sundew.config import SundewConfig
from sundew.core import SundewAlgorithm
from sundew.humanitarian_health import (
    MaternalHealthMonitor,
    MaternalVitals,
    demonstrate_maternal_health_monitoring,
)


def test_maternal_health_comprehensive():
    """Comprehensive maternal health monitoring test."""
    print("COMPREHENSIVE MATERNAL HEALTH MONITORING TEST")
    print("=" * 60)
    print()

    # Test 1: Normal pregnancy monitoring
    print("TEST 1: Normal Pregnancy Monitoring")
    print("-" * 40)

    monitor = MaternalHealthMonitor("TEST_PATIENT_001", gestational_age_weeks=28)

    normal_vitals = [
        MaternalVitals(time.time(), 110, 70, 75, 98, 140, 1.5, 37.0, 10),
        MaternalVitals(time.time() + 300, 115, 72, 78, 99, 145, 1.8, 37.1, 8),
        MaternalVitals(time.time() + 600, 112, 71, 76, 98, 142, 1.6, 36.9, 9),
    ]

    alerts = []
    for vitals in normal_vitals:
        alert = monitor.process_vitals(vitals)
        if alert:
            alerts.append(alert)
            print(f"   [WARNING]  {alert.severity.upper()}: {alert.condition}")
        else:
            print(f"   [OK] Normal vitals: BP {vitals.systolic_bp}/{vitals.diastolic_bp}, "
                      f"HR {vitals.heart_rate}")

    print(f"   Result: {len(alerts)} alerts generated (expected: 0)")
    print()

    # Test 2: Preeclampsia detection
    print("TEST 2: Preeclampsia Detection")
    print("-" * 40)

    preeclampsia_monitor = MaternalHealthMonitor("TEST_PATIENT_002", gestational_age_weeks=34)

    preeclampsia_vitals = [
        MaternalVitals(time.time(), 125, 82, 80, 97, 148, 2.0, 37.2, 7),  # Elevated
        MaternalVitals(time.time() + 300, 145, 95, 85, 96, 152, 2.5, 37.4, 5),  # Preeclampsia
        MaternalVitals(time.time() + 600, 170, 110, 90, 94, 165, 3.2, 37.8, 3),  # Severe
    ]

    preeclampsia_alerts = []
    for i, vitals in enumerate(preeclampsia_vitals):
        alert = preeclampsia_monitor.process_vitals(vitals)
        if alert:
            preeclampsia_alerts.append(alert)
            print(f"   [ALERT] Reading {i+1}: {alert.severity.upper()} - {alert.condition}")
            print(f"      BP: {vitals.systolic_bp}/{vitals.diastolic_bp} mmHg")
            print(f"      Action: {alert.recommended_action}")
        else:
            print(f"   [DATA] Reading {i+1}: Normal "
                      f"(BP {vitals.systolic_bp}/{vitals.diastolic_bp})")
        print()

    print(f"   Result: {len(preeclampsia_alerts)} alerts generated")
    critical_alerts = sum(1 for a in preeclampsia_alerts if a.severity == "critical")
    print(f"   Critical alerts: {critical_alerts}")
    print()

    # Test 3: Energy efficiency analysis
    print("TEST 3: Energy Efficiency Analysis")
    print("-" * 40)

    efficiency_monitor = MaternalHealthMonitor("TEST_PATIENT_003", gestational_age_weeks=30)

    # Simulate 24 hours of monitoring (144 readings, 10min intervals)
    readings_count = 50  # Reduced for demo
    energy_start = efficiency_monitor.sundew.report()["energy_remaining"]

    for i in range(readings_count):
        # Mix of normal and concerning vitals
        if i % 10 == 9:  # 10% concerning readings
            vitals = MaternalVitals(
                time.time() + i * 600,
                145 + (i % 3) * 5, 95, 85, 96, 155, 2.5, 37.3, 4
            )
        else:
            vitals = MaternalVitals(
                time.time() + i * 600,
                115 + (i % 3) * 2, 75, 78, 98, 142, 1.8, 37.0, 8
            )

        efficiency_monitor.process_vitals(vitals)

    energy_end = efficiency_monitor.sundew.report()["energy_remaining"]
    energy_used = energy_start - energy_end

    report = efficiency_monitor.generate_health_report()

    print(f"   Readings processed: {readings_count}")
    print(f"   Energy used: {energy_used:.1f} units")
    print(f"   Energy remaining: {energy_end:.1f}%")
    print(f"   Estimated days remaining: "
          f"{report['system_performance']['estimated_days_remaining']:.1f}")
    print(f"   Energy efficiency: {report['system_performance']['energy_efficiency_pct']:.1f}%")
    print(f"   Lives potentially saved: {report['medical_summary']['lives_potentially_saved']}")
    print()

    # Test 4: Fetal distress detection
    print("TEST 4: Fetal Distress Detection")
    print("-" * 40)

    fetal_monitor = MaternalHealthMonitor("TEST_PATIENT_004", gestational_age_weeks=36)

    fetal_distress_scenarios = [
        MaternalVitals(time.time(), 120, 78, 82, 98, 95, 2.0, 37.1, 6),   # Fetal bradycardia
        MaternalVitals(time.time() + 300, 118, 76, 80, 97, 180, 2.2, 37.0, 8),  # Fetal tachycardia
        MaternalVitals(time.time() + 600, 115, 75, 78, 98, 145, 1.8, 37.0, 10),  # Normal recovery
    ]

    fetal_alerts = []
    for i, vitals in enumerate(fetal_distress_scenarios):
        alert = fetal_monitor.process_vitals(vitals)
        if alert:
            fetal_alerts.append(alert)
            print(f"   [ALERT] Fetal HR {vitals.fetal_heart_rate} bpm: {alert.condition}")
            print(f"      {alert.recommended_action}")
        else:
            print(f"   [OK] Normal fetal HR: {vitals.fetal_heart_rate} bpm")
        print()

    print(f"   Fetal distress alerts: {len(fetal_alerts)}")
    print()

    return {
        "normal_test": len(alerts) == 0,
        "preeclampsia_test": len(preeclampsia_alerts) >= 2,
        "energy_efficiency": report['system_performance']['energy_efficiency_pct'] > 30,
        "fetal_monitoring": len(fetal_alerts) >= 1,
    }


def test_cardiac_monitoring():
    """Test cardiac monitoring capabilities."""
    print("[CARDIAC]  CARDIAC MONITORING TEST")
    print("=" * 40)
    print()

    # Configure for cardiac monitoring
    cardiac_config = SundewConfig(
        activation_threshold=0.4,
        target_activation_rate=0.08,  # 8% for cardiac events
        gate_temperature=0.06,
        max_energy=500.0,
        ema_alpha=0.1,
        adapt_kp=0.01,
        adapt_ki=0.003,
        significance_weights={
            "w_magnitude": 0.4,     # ECG amplitude changes
            "w_anomaly": 0.3,       # Rhythm abnormalities
            "w_context": 0.2,       # Patient history
            "w_urgency": 0.1,       # Immediate risk
        }
    )

    cardiac_monitor = SundewAlgorithm(cardiac_config)

    # Simulate ECG events
    cardiac_events = [
        # Normal sinus rhythm
        {"magnitude": 25, "anomaly_score": 0.1, "context_relevance": 0.5, "urgency": 0.1},
        {"magnitude": 30, "anomaly_score": 0.15, "context_relevance": 0.5, "urgency": 0.1},

        # Arrhythmia events
        # Critical
        {"magnitude": 65, "anomaly_score": 0.8, "context_relevance": 0.9, "urgency": 0.7},
        # Concerning
        {"magnitude": 45, "anomaly_score": 0.6, "context_relevance": 0.7, "urgency": 0.5},

        # Normal variants
        {"magnitude": 35, "anomaly_score": 0.2, "context_relevance": 0.6, "urgency": 0.2},
        {"magnitude": 28, "anomaly_score": 0.12, "context_relevance": 0.5, "urgency": 0.1},

        # Emergency - cardiac arrest pattern
        {"magnitude": 85, "anomaly_score": 0.95, "context_relevance": 1.0, "urgency": 1.0},
    ]

    activations = 0
    critical_events = 0

    for i, event in enumerate(cardiac_events):
        result = cardiac_monitor.process(event)

        if result:
            activations += 1
            severity = "CRITICAL" if event["urgency"] > 0.6 else "WARNING"

            if severity == "CRITICAL":
                critical_events += 1

            print(f"   [ALERT] {severity}: ECG event {i+1} activated")
            print(f"      Magnitude: {event['magnitude']}, Anomaly: {event['anomaly_score']:.2f}")
            print(f"      Urgency: {event['urgency']:.2f}, Significance: {result.significance:.3f}")
        else:
            print(f"   [OK] Normal: ECG event {i+1} (magnitude: {event['magnitude']})")
        print()

    report = cardiac_monitor.report()

    print("CARDIAC MONITORING RESULTS:")
    print(f"   Total events: {len(cardiac_events)}")
    print(f"   Activations: {activations}")
    print(f"   Critical events detected: {critical_events}")
    print(f"   Activation rate: {report['activation_rate']:.1%}")
    print(f"   Energy remaining: {report['energy_remaining']:.1f}")
    print(f"   Energy savings: {report['estimated_energy_savings_pct']:.1f}%")
    print()

    return {
        "cardiac_sensitivity": critical_events >= 2,  # Should detect both critical events
        "cardiac_specificity": activations <= 4,     # Should not over-activate
        "energy_efficiency": report['estimated_energy_savings_pct'] > 50,
    }


def run_comprehensive_demo():
    """Run the comprehensive demonstration."""
    print("[SUNDEW] SUNDEW HEALTH MONITORING - COMPREHENSIVE DEMO")
    print("=" * 60)
    print()

    print("This demo showcases life-saving health monitoring capabilities:")
    print("• Maternal health monitoring for developing countries")
    print("• Cardiac monitoring and arrhythmia detection")
    print("• Ultra-low power consumption for remote deployment")
    print("• Real-time critical alerts with medical recommendations")
    print()

    input("Press Enter to start the maternal health demonstration...")
    print()

    # Run the built-in maternal health demo
    demonstrate_maternal_health_monitoring()

    print()
    input("Press Enter to run comprehensive tests...")
    print()

    # Run comprehensive tests
    maternal_results = test_maternal_health_comprehensive()
    cardiac_results = test_cardiac_monitoring()

    # Summary
    print("[RESULTS] TEST SUMMARY")
    print("=" * 20)

    all_tests = {**maternal_results, **cardiac_results}
    passed = sum(all_tests.values())
    total = len(all_tests)

    print(f"Tests passed: {passed}/{total}")
    print()

    print("Maternal Health Tests:")
    for test, result in maternal_results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test}: {status}")

    print()
    print("Cardiac Monitoring Tests:")
    for test, result in cardiac_results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test}: {status}")

    print()

    if passed == total:
        print("[SUCCESS] ALL TESTS PASSED - System ready for life-saving deployment!")
        print()
        print("Next steps:")
        print("1. Deploy in field trials with healthcare partners")
        print("2. Scale to underserved populations globally")
        print("3. Integrate with existing maternal health programs")
        print("4. Train community health workers on system use")
    else:
        print("[WARNING]  Some tests failed - review system configuration")

    return all_tests


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Health Monitoring Test Guide")
    parser.add_argument(
        "--test",
        choices=["maternal", "cardiac", "all"],
        default="all",
        help="Which test to run"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run comprehensive demo"
    )

    args = parser.parse_args()

    if args.demo:
        run_comprehensive_demo()
    elif args.test == "maternal":
        test_maternal_health_comprehensive()
    elif args.test == "cardiac":
        test_cardiac_monitoring()
    elif args.test == "all":
        print("Running all health monitoring tests...")
        print()
        maternal_results = test_maternal_health_comprehensive()
        cardiac_results = test_cardiac_monitoring()

        all_results = {**maternal_results, **cardiac_results}
        passed = sum(all_results.values())
        total = len(all_results)

        print(f"[RESULTS] FINAL RESULTS: {passed}/{total} tests passed")


if __name__ == "__main__":
    main()
