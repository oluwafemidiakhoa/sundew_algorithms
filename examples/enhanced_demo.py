#!/usr/bin/env python3
"""
Enhanced Sundew Algorithm Demonstration

This script demonstrates the new modular architecture with:
- Neural significance models with attention
- Model Predictive Control (MPC) 
- Real-time monitoring and visualization
- Multi-domain benchmarking
- Statistical analysis

Usage:
    python examples/enhanced_demo.py --mode [basic|neural|mpc|benchmark|monitor]
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sundew.enhanced_core import EnhancedSundewAlgorithm, EnhancedSundewConfig
from sundew.significance_models import LinearSignificanceConfig, NeuralSignificanceConfig
from sundew.gating_strategies import TemperatureGatingConfig, AdaptiveGatingConfig
from sundew.control_policies import PIControlConfig, MPCControlConfig
from sundew.energy_models import SimpleEnergyConfig, RealisticEnergyConfig
from sundew.benchmarking import BenchmarkRunner, BenchmarkConfig
from sundew.monitoring import setup_live_monitoring, setup_monitoring_with_alerts


def demo_basic_enhanced():
    """Demonstrate basic enhanced algorithm with linear significance model."""
    print("=== Basic Enhanced Sundew Demo ===")
    
    # Configure enhanced algorithm with linear significance
    config = EnhancedSundewConfig(
        significance_model="linear",
        gating_strategy="temperature", 
        control_policy="pi",
        energy_model="simple",
        target_activation_rate=0.15,
        component_configs={
            "significance_model": {
                "w_magnitude": 0.3,
                "w_anomaly": 0.4,
                "w_context": 0.2, 
                "w_urgency": 0.1,
                "noise_std": 0.01
            },
            "control_policy": {
                "kp": 0.08,
                "ki": 0.02,
                "adaptive_gains": True,
                "stability_monitoring": True
            }
        }
    )
    
    # Create algorithm
    algorithm = EnhancedSundewAlgorithm(config)
    
    print("Processing sample data stream...")
    
    # Generate sample data
    results = []
    for i in range(100):
        # Simulate varying input patterns
        if i < 20:
            # High activity period
            sample = {
                "magnitude": 70 + (i % 10) * 3,
                "anomaly_score": 0.7 + (i % 5) * 0.05,
                "context_relevance": 0.8,
                "urgency": 0.6
            }
        elif i < 50:
            # Normal activity
            sample = {
                "magnitude": 45 + (i % 8) * 2,
                "anomaly_score": 0.3 + (i % 4) * 0.1,
                "context_relevance": 0.5,
                "urgency": 0.2
            }
        else:
            # Low activity period  
            sample = {
                "magnitude": 25 + (i % 6) * 1,
                "anomaly_score": 0.1 + (i % 3) * 0.05,
                "context_relevance": 0.3,
                "urgency": 0.1
            }
        
        result = algorithm.process(sample)
        results.append({
            "activated": result is not None,
            "significance": result.significance if result else 0.0,
            "energy": result.energy_consumed if result else 0.0,
            "threshold": algorithm.control_state.threshold
        })
        
        # Print progress every 20 samples
        if (i + 1) % 20 == 0:
            report = algorithm.get_comprehensive_report()
            print(f"  Sample {i+1}: Rate={report['activation_rate']:.3f}, "
                  f"Energy={report['energy_remaining']:.1f}, "
                  f"Threshold={report['current_threshold']:.3f}")
    
    # Final report
    final_report = algorithm.get_comprehensive_report()
    print("\n=== Final Results ===")
    print(f"Activation Rate: {final_report['activation_rate']:.3f}")
    print(f"Energy Efficiency: {final_report['energy_efficiency']:.1%}")
    print(f"Research Quality Score: {final_report['research_quality_score']:.1f}/10")
    print(f"Performance Score: {final_report['performance_score']:.1f}/10")
    
    return algorithm, results


def demo_neural_significance():
    """Demonstrate neural significance model with temporal attention."""
    print("\n=== Neural Significance Model Demo ===")
    
    config = EnhancedSundewConfig(
        significance_model="neural",
        gating_strategy="adaptive",
        control_policy="pi",
        energy_model="simple",
        enable_online_learning=True,
        component_configs={
            "significance_model": {
                "hidden_sizes": [32, 16],
                "learning_rate": 0.001,
                "temporal_window": 10,
                "attention_heads": 4,
                "use_temporal_attention": True
            },
            "gating_strategy": {
                "attention_dim": 16,
                "context_window": 20,
                "confidence_threshold": 0.8,
                "multi_objective_weights": {
                    "accuracy": 0.5,
                    "energy": 0.3,
                    "latency": 0.2
                }
            }
        }
    )
    
    algorithm = EnhancedSundewAlgorithm(config)
    
    print("Training neural model with temporal patterns...")
    
    # Generate sequential data with temporal dependencies
    activations = []
    for i in range(200):
        # Create temporal patterns - events tend to cluster
        if i > 10:
            recent_activations = sum(activations[-10:])
            base_prob = 0.3 if recent_activations > 3 else 0.1
        else:
            base_prob = 0.2
        
        is_event = (i + hash(str(i)) % 100) / 100.0 < base_prob
        
        if is_event:
            sample = {
                "magnitude": 75 + (i % 7) * 4,
                "anomaly_score": 0.8 + (i % 3) * 0.05,
                "context_relevance": 0.7,
                "urgency": 0.8
            }
            ground_truth = 1
        else:
            sample = {
                "magnitude": 35 + (i % 5) * 2,
                "anomaly_score": 0.2 + (i % 4) * 0.05,
                "context_relevance": 0.4,
                "urgency": 0.2
            }
            ground_truth = 0
        
        result = algorithm.process(sample)
        activated = 1 if result else 0
        activations.append(activated)
        
        # Provide feedback for learning (simulate outcome quality)
        if result and algorithm.config.enable_online_learning:
            # Simulate processing outcome feedback
            accuracy = 0.9 if ground_truth == activated else 0.1
            energy_efficiency = 0.8
            outcome = {
                "accuracy": accuracy,
                "energy_efficiency": energy_efficiency,
                "processing_time": result.processing_time
            }
            
            # Update models (this is normally done internally)
            from sundew.interfaces import ProcessingContext
            context = ProcessingContext(
                timestamp=time.time(),
                sequence_id=i,
                features=sample,
                history=[],
                metadata={}
            )
    
    final_report = algorithm.get_comprehensive_report()
    print(f"\nNeural Model Results:")
    print(f"Activation Rate: {final_report['activation_rate']:.3f}")
    print(f"Research Quality Score: {final_report['research_quality_score']:.1f}/10")
    
    # Show neural model details
    neural_params = final_report['components']['significance_model']['parameters']
    print(f"Neural Network Layers: {len(neural_params['weights'])}")
    print(f"Temporal Attention: {'Enabled' if neural_params['config']['use_temporal_attention'] else 'Disabled'}")
    
    return algorithm


def demo_mpc_control():
    """Demonstrate Model Predictive Control with stability analysis."""
    print("\n=== Model Predictive Control Demo ===")
    
    config = EnhancedSundewConfig(
        significance_model="linear",
        gating_strategy="temperature",
        control_policy="mpc",
        energy_model="realistic",
        target_activation_rate=0.12,
        component_configs={
            "control_policy": {
                "prediction_horizon": 20,
                "control_horizon": 10,
                "weight_tracking": 1.0,
                "weight_control_effort": 0.1,
                "weight_energy": 0.5
            },
            "energy_model": {
                "platform": "cortex_m4",
                "battery_capacity_mah": 2000,
                "thermal_throttling_threshold": 70.0,
                "dvfs_enabled": True
            }
        }
    )
    
    algorithm = EnhancedSundewAlgorithm(config)
    
    print("Running MPC with realistic energy model...")
    
    # Test MPC's handling of disturbances and constraints
    threshold_history = []
    energy_history = []
    stability_metrics = []
    
    for i in range(150):
        # Add periodic disturbances to test robustness
        disturbance = 0.1 * (i % 30 - 15) / 15  # Sine-like disturbance
        
        sample = {
            "magnitude": 50 + 20 * disturbance,
            "anomaly_score": 0.4 + 0.2 * abs(disturbance),
            "context_relevance": 0.5,
            "urgency": 0.3 + 0.1 * disturbance
        }
        
        result = algorithm.process(sample)
        
        # Track system evolution
        threshold_history.append(algorithm.control_state.threshold)
        energy_history.append(algorithm.control_state.energy_level)
        stability_metrics.append(algorithm.control_state.stability_metrics)
        
        if (i + 1) % 30 == 0:
            # Get MPC predictions
            mpc_policy = algorithm.control_policy
            stability_pred = mpc_policy.predict_stability(algorithm.control_state)
            
            print(f"  Step {i+1}: Threshold={algorithm.control_state.threshold:.3f}, "
                  f"Predicted Settling Time={stability_pred['convergence_time']:.1f}, "
                  f"Stability Margin={stability_pred['stability_margin']:.3f}")
    
    final_report = algorithm.get_comprehensive_report()
    mpc_metrics = final_report['components']['control_policy']
    
    print(f"\nMPC Control Results:")
    print(f"Convergence Time: {mpc_metrics['stability_prediction']['convergence_time']:.1f}")
    print(f"Overshoot: {mpc_metrics['stability_prediction']['predicted_overshoot']:.3f}")
    print(f"Stability Margin: {mpc_metrics['stability_prediction']['stability_margin']:.3f}")
    
    # Analyze stability over time
    import numpy as np
    threshold_stability = np.std(threshold_history[-50:])  # Last 50 samples
    print(f"Threshold Stability (std): {threshold_stability:.4f}")
    
    return algorithm, {"thresholds": threshold_history, "energy": energy_history}


def demo_comprehensive_benchmark():
    """Demonstrate multi-domain benchmarking with statistical analysis."""
    print("\n=== Comprehensive Benchmarking Demo ===")
    
    # Configure benchmark
    benchmark_config = BenchmarkConfig(
        num_seeds=3,  # Reduced for demo
        num_samples=5000,
        confidence_level=0.95,
        include_baselines=True,
        baseline_strategies=["random", "fixed_threshold", "oracle"],
        save_detailed_logs=True,
        output_directory="demo_benchmark_results"
    )
    
    # Create algorithm configurations to compare
    algorithm_configs = [
        ("linear_pi", EnhancedSundewConfig(
            significance_model="linear",
            control_policy="pi",
            target_activation_rate=0.15
        )),
        ("neural_adaptive", EnhancedSundewConfig(
            significance_model="neural",
            gating_strategy="adaptive",
            control_policy="pi",
            enable_online_learning=True,
            component_configs={
                "significance_model": {"use_temporal_attention": True}
            }
        )),
        ("linear_mpc", EnhancedSundewConfig(
            significance_model="linear",
            control_policy="mpc",
            target_activation_rate=0.15,
            component_configs={
                "control_policy": {"prediction_horizon": 15}
            }
        ))
    ]
    
    # Run benchmark
    runner = BenchmarkRunner(benchmark_config)
    print("Running benchmark across ECG, Vision, and Audio domains...")
    
    results = runner.run_comprehensive_benchmark(algorithm_configs)
    
    # Display key results
    print("\n=== Benchmark Results Summary ===")
    
    for domain, domain_results in results["domain_results"].items():
        print(f"\n{domain.upper()} Domain:")
        
        # Compare algorithm performance
        for algo_name, algo_results in domain_results.items():
            if algo_results and algo_name not in benchmark_config.baseline_strategies:
                avg_f1 = np.mean([r.f1_score for r in algo_results])
                avg_energy = np.mean([r.energy_efficiency for r in algo_results])
                print(f"  {algo_name:15} - F1: {avg_f1:.3f}, Energy: {avg_energy:.1%}")
        
        # Best performers
        best = results["summary"]["best_performers_by_domain"][domain]
        print(f"  Best F1: {best['best_f1'][0]} ({best['best_f1'][1]:.3f})")
        print(f"  Best Energy: {best['best_energy'][0]} ({best['best_energy'][1]:.1%})")
    
    # Overall research quality improvement
    quality_assessment = results["summary"]["research_quality_assessment"]
    print(f"\nResearch Quality Assessment:")
    print(f"  Baseline Score: {quality_assessment['baseline_score']:.1f}/10")
    print(f"  Current Score: {quality_assessment['estimated_current_score']:.1f}/10")
    print(f"  Improvement: +{quality_assessment['improvement']:.1f} points")
    print(f"  Target Score: {quality_assessment['target_score']:.1f}/10")
    
    return results


def demo_live_monitoring():
    """Demonstrate real-time monitoring with alerts."""
    print("\n=== Real-Time Monitoring Demo ===")
    
    # Setup algorithm with monitoring-friendly configuration
    config = EnhancedSundewConfig(
        significance_model="neural",
        gating_strategy="adaptive", 
        control_policy="mpc",
        energy_model="realistic",
        enable_online_learning=True
    )
    
    algorithm = EnhancedSundewAlgorithm(config)
    
    # Setup monitoring with custom alert handler
    def alert_handler(alert_type: str, alert_data: dict):
        severity_emojis = {
            "LOW": "🟡",
            "MEDIUM": "🟠", 
            "HIGH": "🔴",
            "CRITICAL": "🚨"
        }
        emoji = severity_emojis.get(alert_data.get("severity", "LOW"), "⚠️")
        
        print(f"{emoji} ALERT: {alert_data['metric']} = {alert_data['value']:.3f} "
              f"(threshold: {alert_data.get('threshold', 'N/A')})")
    
    monitor = setup_monitoring_with_alerts(algorithm, alert_handler)
    
    print("Running monitored processing with synthetic data...")
    print("Watch for alerts and performance metrics...")
    
    # Simulate realistic processing with various conditions
    scenarios = [
        {"name": "Normal Operation", "samples": 50, "base_magnitude": 45},
        {"name": "High Activity Burst", "samples": 30, "base_magnitude": 80}, 
        {"name": "Low Energy Scenario", "samples": 40, "base_magnitude": 35},
        {"name": "Unstable Conditions", "samples": 30, "base_magnitude": 60}
    ]
    
    sample_count = 0
    predictions = []
    ground_truth = []
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        
        for i in range(scenario["samples"]):
            # Generate sample based on scenario
            base_mag = scenario["base_magnitude"]
            
            # Add scenario-specific variations
            if "Burst" in scenario["name"]:
                magnitude = base_mag + 20 * (1 + 0.3 * (i % 5))
                is_positive = True
            elif "Low Energy" in scenario["name"]:
                magnitude = base_mag - 10 * (i / scenario["samples"])
                is_positive = (i % 10) < 2  # Sparse events
            elif "Unstable" in scenario["name"]:
                magnitude = base_mag + 30 * (-1 if i % 6 < 3 else 1)  # Oscillating
                is_positive = magnitude > 50
            else:  # Normal
                magnitude = base_mag + 15 * np.random.normal(0, 1)
                is_positive = magnitude > 55
            
            sample = {
                "magnitude": max(0, magnitude),
                "anomaly_score": 0.8 if is_positive else 0.2,
                "context_relevance": 0.6 if is_positive else 0.3,
                "urgency": 0.7 if is_positive else 0.2
            }
            
            # Process and monitor
            result = algorithm.process(sample)
            prediction = 1 if result else 0
            
            predictions.append(prediction)
            ground_truth.append(1 if is_positive else 0)
            
            # Update monitor with ground truth for performance tracking
            monitor.update(algorithm, 1 if is_positive else 0, prediction)
            
            sample_count += 1
            
            # Brief pause to simulate real-time processing
            time.sleep(0.01)
        
        # Print scenario summary
        scenario_report = algorithm.get_comprehensive_report()
        print(f"  Scenario complete - Rate: {scenario_report['activation_rate']:.3f}, "
              f"Energy: {scenario_report['energy_remaining']:.1f}")
    
    # Final monitoring summary
    monitoring_summary = monitor.get_monitoring_summary()
    print(f"\n=== Monitoring Summary ===")
    print(f"Samples Monitored: {monitoring_summary['samples_collected']}")
    print(f"System Health: {monitoring_summary['system_health']['status']} "
          f"({monitoring_summary['system_health']['overall_score']:.1f}/100)")
    print(f"Recent Alerts: {monitoring_summary['alerts']['total_recent']}")
    print(f"Critical Alerts: {monitoring_summary['alerts']['critical']}")
    
    # Export monitoring data
    monitor.export_data("demo_monitoring_data.json")
    print("Monitoring data exported to: demo_monitoring_data.json")
    
    monitor.stop_monitoring()
    return algorithm, monitor


def main():
    """Main demonstration runner."""
    parser = argparse.ArgumentParser(description="Enhanced Sundew Algorithm Demo")
    parser.add_argument("--mode", choices=["basic", "neural", "mpc", "benchmark", "monitor", "all"],
                       default="basic", help="Demo mode to run")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    print("🌿 Enhanced Sundew Algorithm Demonstration")
    print("==========================================")
    
    try:
        if args.mode == "basic" or args.mode == "all":
            demo_basic_enhanced()
        
        if args.mode == "neural" or args.mode == "all":
            demo_neural_significance()
        
        if args.mode == "mpc" or args.mode == "all":
            demo_mpc_control()
        
        if args.mode == "benchmark" or args.mode == "all":
            demo_comprehensive_benchmark()
        
        if args.mode == "monitor" or args.mode == "all":
            demo_live_monitoring()
        
        print("\n✅ Demo completed successfully!")
        print("\nNext steps:")
        print("1. Explore the modular components in src/sundew/")
        print("2. Run benchmarks with your own data")
        print("3. Customize configurations for your domain")
        print("4. Set up monitoring for production deployment")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    # Ensure numpy is available
    try:
        import numpy as np
    except ImportError:
        print("Error: numpy is required for the enhanced demos")
        print("Install with: uv add numpy")
        sys.exit(1)
    
    sys.exit(main())