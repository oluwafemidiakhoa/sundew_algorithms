#!/usr/bin/env python3
"""
Quick performance test for enhanced Sundew system.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sundew.config import SundewConfig
from sundew.core import SundewAlgorithm
from sundew.enhanced_core import EnhancedSundewAlgorithm, EnhancedSundewConfig


def generate_test_data(n_samples=1000, seed=42):
    """Generate realistic test data for benchmarking."""
    np.random.seed(seed)

    samples = []
    for i in range(n_samples):
        # Create realistic sensor data patterns
        base_magnitude = 45 + 30 * np.sin(i * 0.02) + 5 * np.random.normal()

        # Add anomalies occasionally
        is_anomaly = np.random.random() < 0.1
        anomaly_boost = 25 if is_anomaly else 0

        sample = {
            "magnitude": max(0, base_magnitude + anomaly_boost),
            "anomaly_score": (0.8 if is_anomaly else 0.2) + 0.1 * np.random.normal(),
            "context_relevance": 0.6 + 0.3 * np.sin(i * 0.01) + 0.1 * np.random.normal(),
            "urgency": 0.3 + (0.4 if is_anomaly else 0) + 0.1 * np.random.normal()
        }

        # Clamp values to valid ranges
        for key in sample:
            if key != "magnitude":
                sample[key] = np.clip(sample[key], 0.0, 1.0)

        samples.append(sample)

    return samples


def benchmark_original_system(samples, verbose=False):
    """Benchmark original Sundew system."""
    print("Testing Original Sundew System...")

    config = SundewConfig(
        activation_threshold=0.65,
        target_activation_rate=0.15,
        gate_temperature=0.1,
        rng_seed=42
    )

    algorithm = SundewAlgorithm(config)

    start_time = time.time()
    results = []

    for i, sample in enumerate(samples):
        result = algorithm.process(sample)
        if result:
            results.append({
                "activated": True,
                "significance": result.significance,
                "energy_consumed": result.energy_consumed
            })
        else:
            results.append({
                "activated": False,
                "significance": 0.0,
                "energy_consumed": 0.5  # Dormant cost
            })

        if verbose and i % 100 == 0:
            print(f"  Processed {i+1}/{len(samples)} samples")

    processing_time = time.time() - start_time
    final_report = algorithm.report()

    return {
        "system": "original",
        "samples_processed": len(samples),
        "processing_time": processing_time,
        "throughput": len(samples) / processing_time,
        "activation_rate": final_report["activation_rate"],
        "energy_remaining": final_report["energy_remaining"],
        "energy_savings": final_report["estimated_energy_savings_pct"],
        "results": results[:10]  # Sample of results
    }


def benchmark_enhanced_system(samples, model_type="linear", control_type="pi", verbose=False):
    """Benchmark enhanced Sundew system."""
    print(f"Testing Enhanced Sundew System ({model_type} + {control_type})...")

    config = EnhancedSundewConfig(
        significance_model=model_type,  # "linear" or "neural"
        gating_strategy="temperature",
        control_policy=control_type,    # "pi" or "mpc"
        energy_model="realistic",
        target_activation_rate=0.15,
        initial_threshold=0.65,
        enable_online_learning=True
    )

    algorithm = EnhancedSundewAlgorithm(config)

    start_time = time.time()
    results = []

    for i, sample in enumerate(samples):
        result = algorithm.process(sample)
        results.append({
            "activated": result.activated,
            "significance": result.significance,
            "energy_consumed": result.energy_consumed,
            "processing_time": result.processing_time
        })

        if verbose and i % 100 == 0:
            print(f"  Processed {i+1}/{len(samples)} samples")

    processing_time = time.time() - start_time
    final_report = algorithm.get_comprehensive_report()

    return {
        "system": f"enhanced_{model_type}_{control_type}",
        "samples_processed": len(samples),
        "processing_time": processing_time,
        "throughput": len(samples) / processing_time,
        "activation_rate": final_report["activation_rate"],
        "energy_remaining": final_report["energy_remaining"],
        "energy_savings": final_report.get("energy_efficiency", 0) * 100,
        "research_quality_score": final_report["research_quality_score"],
        "stability_metrics": final_report["stability_metrics"],
        "component_performance": final_report["components"],
        "total_samples": final_report["total_samples"],
        "activations": final_report["activations"],
        "results": results[:10]  # Sample of results
    }


def main():
    """Run comprehensive performance comparison."""
    print("Enhanced Sundew Performance Benchmark")
    print("=" * 50)

    # Generate test data
    print("Generating test data...")
    samples = generate_test_data(500, seed=42)  # Smaller sample for quick test
    print(f"Generated {len(samples)} test samples")

    # Test systems
    benchmark_results = []

    try:
        # Original system
        original_result = benchmark_original_system(samples, verbose=True)
        benchmark_results.append(original_result)
        print(f"Original: {original_result['throughput']:.1f} samples/sec, "
              f"{original_result['energy_savings']:.1f}% energy savings")

    except Exception as e:
        print(f"Original system error: {e}")

    try:
        # Enhanced system variants
        for model_type in ["linear", "neural"]:
            for control_type in ["pi", "mpc"]:
                try:
                    enhanced_result = benchmark_enhanced_system(
                        samples, model_type, control_type, verbose=True
                    )
                    benchmark_results.append(enhanced_result)
                    print(f"Enhanced ({model_type}+{control_type}): "
                          f"{enhanced_result['throughput']:.1f} samples/sec, "
                          f"Research Quality: {enhanced_result['research_quality_score']:.1f}/10")

                except Exception as e:
                    print(f"Enhanced {model_type}+{control_type} error: {e}")

    except Exception as e:
        print(f"Enhanced systems error: {e}")

    # Save results
    results_file = "results/enhanced_performance_comparison.json"
    Path("results").mkdir(exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump({
            "benchmark_metadata": {
                "timestamp": time.time(),
                "samples_tested": len(samples),
                "test_description": (
                    "Comprehensive performance comparison of original vs enhanced Sundew"
                )
            },
            "results": benchmark_results
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Summary table
    print("\nPerformance Summary:")
    print("-" * 80)
    print(f"{'System':<25} {'Throughput':<12} {'Energy Savings':<15} {'Quality Score':<15}")
    print("-" * 80)

    for result in benchmark_results:
        system_name = result["system"]
        throughput = f"{result['throughput']:.1f} smp/s"
        energy_savings = f"{result['energy_savings']:.1f}%"
        quality_score = f"{result.get('research_quality_score', 'N/A')}"

        print(f"{system_name:<25} {throughput:<12} {energy_savings:<15} {quality_score:<15}")

    print("-" * 80)
    print("Benchmark completed successfully!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
