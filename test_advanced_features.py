#!/usr/bin/env python3
"""
Comprehensive testing suite for advanced Sundew algorithm features.

This script tests all advanced features including:
- Information-theoretic threshold adaptation
- High-performance batch processing
- AutoML hyperparameter optimization
- Theoretical analysis and convergence proofs
- Configuration presets

Run this to verify that all advanced features are working correctly.
"""

import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from sundew.enhanced_core import EnhancedSundewAlgorithm, EnhancedSundewConfig
    print("[OK] Enhanced core imported successfully")
except Exception as e:
    print(f"[ERROR] Failed to import enhanced core: {e}")
    sys.exit(1)


def generate_test_data(n_samples: int = 1000) -> List[Dict[str, Any]]:
    """Generate synthetic test data for advanced features testing."""
    np.random.seed(42)

    test_data = []
    for i in range(n_samples):
        # Create realistic data with patterns
        base_magnitude = np.random.exponential(20)

        # Add some structure
        if i % 100 < 20:  # 20% high-importance samples
            base_magnitude *= 2
            anomaly_score = np.random.uniform(0.7, 1.0)
        else:
            anomaly_score = np.random.uniform(0.0, 0.6)

        sample = {
            'magnitude': base_magnitude,
            'anomaly_score': anomaly_score,
            'context_relevance': np.random.uniform(0, 1),
            'urgency': np.random.uniform(0, 1),
            'timestamp': time.time() + i * 0.001,
            'sequence_id': i
        }

        test_data.append(sample)

    return test_data


def test_configuration_presets():
    """Test AutoML-optimized configuration presets."""
    print("Testing configuration presets...")

    # Test preset creation
    presets_info = EnhancedSundewConfig.get_available_presets()
    print(f"  Available domains: {list(presets_info['domains'].keys())}")
    print(f"  Available targets: {list(presets_info['targets'].keys())}")

    # Test each domain preset
    test_results = {}
    for domain in presets_info['domains'].keys():
        try:
            config = EnhancedSundewConfig.create_optimized_config(
                application_domain=domain,
                performance_target="balanced"
            )

            # Verify configuration is valid
            assert 0.01 <= config.target_activation_rate <= 1.0
            assert 0.0 <= config.min_threshold <= config.max_threshold <= 1.0
            assert config.performance_window > 0
            assert config.log_frequency > 0

            test_results[domain] = "PASS"
            print(f"  {domain}: [OK] Configuration created successfully")

        except Exception as e:
            test_results[domain] = f"FAIL: {e}"
            print(f"  {domain}: [ERROR] {e}")

    return test_results


def test_information_theoretic_features(algorithm, test_data):
    """Test information-theoretic threshold adaptation."""
    print("Testing information-theoretic features...")

    if not algorithm.information_controller:
        print("  [SKIP] Information-theoretic controller not enabled")
        return {"status": "skipped"}

    try:
        # Process some data to generate history
        for i, sample in enumerate(test_data[:100]):
            algorithm.process(sample)

        # Get information-theoretic analysis report
        info_report = algorithm.information_controller.get_comprehensive_report()

        # Verify report structure
        required_keys = ['method', 'current_threshold', 'performance_summary']
        for key in required_keys:
            assert key in info_report, f"Missing key: {key}"

        performance = info_report['performance_summary']
        print(f"  Current threshold: {info_report['current_threshold']:.3f}")
        print(f"  Mutual information: {performance['avg_mutual_information']:.4f}")
        print(f"  Entropy: {performance['avg_entropy']:.4f}")
        print(f"  Adaptations: {performance['total_adaptations']}")

        return {"status": "pass", "report": info_report}

    except Exception as e:
        print(f"  [ERROR] Information-theoretic test failed: {e}")
        return {"status": "fail", "error": str(e)}


def test_batch_processing_features(algorithm, test_data):
    """Test high-performance batch processing."""
    print("Testing batch processing features...")

    if not algorithm.batch_engine:
        print("  [SKIP] Batch processing engine not enabled")
        return {"status": "skipped"}

    try:
        # Test individual processing vs batch processing
        batch_size = 500
        test_batch = test_data[:batch_size]

        # Individual processing (baseline)
        start_time = time.time()
        individual_results = []
        for sample in test_batch:
            result = algorithm.process(sample)
            individual_results.append({
                'activated': result.activated,
                'significance': result.significance,
                'energy': result.energy_consumed
            })
        individual_time = time.time() - start_time

        # Batch processing
        start_time = time.time()
        batch_result = algorithm.process_batch(test_batch)
        batch_time = time.time() - start_time

        # Compare results
        speedup = individual_time / batch_time if batch_time > 0 else float('inf')
        individual_activation_rate = np.mean([r['activated'] for r in individual_results])

        print(f"  Individual processing: {individual_time:.3f}s")
        print(f"  Batch processing: {batch_time:.3f}s")
        print(f"  Speedup: {speedup:.1f}x")
        print(f"  Batch activation rate: {batch_result.activation_rate:.1%}")
        print(f"  Individual activation rate: {individual_activation_rate:.1%}")
        print(f"  Throughput: {batch_result.metadata.get('throughput', 0):.0f} samples/sec")

        # Benchmark processors
        benchmark_results = algorithm.benchmark_batch_processors(test_batch[:100])
        if 'error' not in benchmark_results:
            print(f"  Benchmark results: {list(benchmark_results.keys())}")

        return {
            "status": "pass",
            "speedup": speedup,
            "batch_result": batch_result,
            "benchmark": benchmark_results
        }

    except Exception as e:
        print(f"  [ERROR] Batch processing test failed: {e}")
        return {"status": "fail", "error": str(e)}


def test_automl_optimization(algorithm, test_data):
    """Test AutoML hyperparameter optimization."""
    print("Testing AutoML optimization...")

    if not algorithm.automl_optimizer:
        print("  [SKIP] AutoML optimizer not enabled")
        return {"status": "skipped"}

    try:
        # Run short optimization
        print("  Running AutoML optimization (this may take a moment)...")
        optimal_params = algorithm.optimize_with_automl(
            test_data=test_data[:200],  # Small subset for speed
            time_budget_minutes=1  # Very short for testing
        )

        if optimal_params:
            print("  Optimization successful!")
            print(f"  Found {len(optimal_params)} optimal parameters")

            # Show a few key parameters
            key_params = ['target_activation_rate', 'initial_threshold', 'adapt_kp']
            for param in key_params:
                if param in optimal_params:
                    print(f"    {param}: {optimal_params[param]:.4f}")
        else:
            print("  [WARNING] Optimization returned no results")

        return {"status": "pass", "optimal_params": optimal_params}

    except Exception as e:
        print(f"  [ERROR] AutoML optimization test failed: {e}")
        return {"status": "fail", "error": str(e)}


def test_theoretical_analysis(algorithm, test_data):
    """Test theoretical analysis and convergence proofs."""
    print("Testing theoretical analysis...")

    if not algorithm.theoretical_analyzer:
        print("  [SKIP] Theoretical analysis engine not enabled")
        return {"status": "skipped"}

    try:
        # Process data to generate metrics history
        for sample in test_data[:200]:
            algorithm.process(sample)

        # Get comprehensive report (includes theoretical analysis)
        report = algorithm.get_comprehensive_report()

        if "theoretical_analysis" in report:
            theoretical = report["theoretical_analysis"]

            if "error" in theoretical:
                print(f"  [WARNING] Theoretical analysis error: {theoretical['error']}")
                return {"status": "warning", "error": theoretical['error']}

            # Check analysis components
            if "stability_analysis" in theoretical:
                stability = theoretical["stability_analysis"]
                pi_stability = stability.get("pi_controller", {})

                print(f"  PI Controller Stability: {pi_stability.get('converged', 'Unknown')}")
                print(f"  Convergence Rate: {pi_stability.get('convergence_rate', 0):.4f}")
                print(f"  Settling Time: {pi_stability.get('settling_time', 'inf')} seconds")

            if "performance_bounds" in theoretical:
                bounds = theoretical["performance_bounds"]
                energy_bounds = bounds.get("energy_savings_bounds", (0, 0))
                print(f"  Energy Savings Bounds: {energy_bounds[0]:.1%} - {energy_bounds[1]:.1%}")

            if "theoretical_guarantees" in theoretical:
                guarantees = theoretical["theoretical_guarantees"]
                print(f"  Stability Guaranteed: {guarantees.get('stability_guaranteed', False)}")
                print(f"  Energy Savings Guaranteed: "
                      f"{guarantees.get('energy_savings_guaranteed', False)}")

            return {"status": "pass", "analysis": theoretical}
        else:
            print("  [WARNING] No theoretical analysis in report")
            return {"status": "warning", "message": "No analysis found"}

    except Exception as e:
        print(f"  [ERROR] Theoretical analysis test failed: {e}")
        return {"status": "fail", "error": str(e)}


def test_comprehensive_integration(algorithm, test_data):
    """Test comprehensive integration of all features."""
    print("Testing comprehensive integration...")

    try:
        # Process a meaningful amount of data
        print("  Processing test data...")
        processed_count = 0
        for sample in test_data[:500]:
            algorithm.process(sample)
            processed_count += 1

            if processed_count % 100 == 0:
                print(f"    Processed {processed_count} samples...")

        # Get comprehensive report
        print("  Generating comprehensive report...")
        report = algorithm.get_comprehensive_report()

        # Verify report completeness
        expected_sections = [
            'total_samples', 'activation_rate', 'energy_efficiency',
            'performance_score', 'research_quality_score'
        ]

        missing_sections = []
        for section in expected_sections:
            if section not in report:
                missing_sections.append(section)

        if missing_sections:
            print(f"  [WARNING] Missing report sections: {missing_sections}")

        # Display key metrics
        print("  Final Results:")
        print(f"    Total Samples: {report.get('total_samples', 0)}")
        print(f"    Activation Rate: {report.get('activation_rate', 0):.1%}")
        print(f"    Energy Efficiency: {report.get('energy_efficiency', 0):.1%}")
        print(f"    Performance Score: {report.get('performance_score', 0):.1f}/10")
        print(f"    Research Quality Score: {report.get('research_quality_score', 0):.1f}/10")

        # Check advanced feature integration
        advanced_features = [
            'information_theoretic_analysis',
            'batch_processing_performance',
            'theoretical_analysis'
        ]

        for feature in advanced_features:
            if feature in report:
                status = "ERROR" if "error" in report[feature] else "OK"
                print(f"    {feature}: [{status}]")

        return {
            "status": "pass",
            "report": report,
            "missing_sections": missing_sections
        }

    except Exception as e:
        print(f"  [ERROR] Comprehensive integration test failed: {e}")
        return {"status": "fail", "error": str(e)}


def main():
    """Run comprehensive advanced features test suite."""
    print("Sundew Algorithm Advanced Features Test Suite")
    print("=" * 60)

    # Generate test data
    print("Generating test data...")
    test_data = generate_test_data(1000)
    print(f"Generated {len(test_data)} test samples")
    print()

    # Test configuration presets
    preset_results = test_configuration_presets()
    print()

    # Create research configuration for testing all features
    print("Creating research configuration with all advanced features...")
    try:
        config = EnhancedSundewConfig.create_optimized_config(
            application_domain="research",
            performance_target="balanced"
        )

        # Initialize algorithm
        algorithm = EnhancedSundewAlgorithm(config)
        print("[OK] Advanced algorithm initialized successfully")
        print(
            f"  Information Controller: "
            f"{'Enabled' if algorithm.information_controller else 'Disabled'}"
        )
        print(f"  Batch Engine: {'Enabled' if algorithm.batch_engine else 'Disabled'}")
        print(f"  AutoML Optimizer: {'Enabled' if algorithm.automl_optimizer else 'Disabled'}")
        print(
            f"  Theoretical Analyzer: {'Enabled' if algorithm.theoretical_analyzer else 'Disabled'}"
        )

    except Exception as e:
        print(f"[ERROR] Failed to initialize advanced algorithm: {e}")
        return

    print()

    # Run feature tests
    test_results = {}

    test_results['information_theoretic'] = test_information_theoretic_features(
        algorithm, test_data
    )
    print()

    test_results['batch_processing'] = test_batch_processing_features(algorithm, test_data)
    print()

    test_results['automl'] = test_automl_optimization(algorithm, test_data)
    print()

    test_results['theoretical'] = test_theoretical_analysis(algorithm, test_data)
    print()

    test_results['integration'] = test_comprehensive_integration(algorithm, test_data)
    print()

    # Summary
    print("Test Summary")
    print("-" * 30)

    total_tests = 0
    passed_tests = 0

    # Configuration presets
    for domain, result in preset_results.items():
        total_tests += 1
        if result == "PASS":
            passed_tests += 1
            print(f"Config {domain}: [PASS]")
        else:
            print(f"Config {domain}: [FAIL] {result}")

    # Advanced features
    for feature, result in test_results.items():
        total_tests += 1
        status = result.get('status', 'unknown').upper()
        if status == 'PASS':
            passed_tests += 1
            print(f"{feature}: [PASS]")
        elif status == 'SKIP' or status == 'SKIPPED':
            print(f"{feature}: [SKIP]")
        elif status == 'WARNING':
            print(f"{feature}: [WARNING] {result.get('message', result.get('error', ''))}")
        else:
            print(f"{feature}: [FAIL] {result.get('error', 'Unknown error')}")

    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n[SUCCESS] All advanced features are working correctly!")
        research_quality = (
            test_results.get('integration', {})
            .get('report', {})
            .get('research_quality_score', 0)
        )
        print(f"Research Quality Score: {research_quality:.1f}/10")
    else:
        print(
            f"\n[WARNING] Some features need attention ({total_tests - passed_tests} issues found)"
        )


if __name__ == "__main__":
    main()
