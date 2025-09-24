#!/usr/bin/env python3
"""
Stress test the Sundew algorithm with challenging datasets
Tests video processing, adversarial cases, and edge conditions before git push
"""

import numpy as np
import random
from src.sundew.simple_core import SimpleSundewAlgorithm
from src.sundew.config import SundewConfig
from typing import Dict, List, Tuple

def generate_video_frame_dataset(n_frames: int, scene_change_rate: float = 0.02) -> Tuple[List[Dict], List[int]]:
    """
    Generate video frame processing dataset
    Most frames are similar (low significance), scene changes are high significance
    """
    print(f"Generating video dataset: {n_frames} frames, {scene_change_rate*100:.1f}% scene changes")

    samples = []
    labels = []

    # Simulate video processing where most frames are redundant
    baseline_magnitude = 20
    baseline_motion = 0.1

    random.seed(42)

    for i in range(n_frames):
        is_scene_change = random.random() < scene_change_rate

        if is_scene_change:
            # Scene change - high motion, different content
            sample = {
                "magnitude": random.uniform(80, 100),  # High frame difference
                "anomaly_score": random.uniform(0.8, 1.0),  # Different scene
                "context": random.uniform(0.7, 1.0),  # High motion vectors
                "urgency": random.uniform(0.6, 1.0)  # Significant change
            }
            labels.append(1)
        else:
            # Normal frame - low change from previous
            sample = {
                "magnitude": baseline_magnitude + random.uniform(-5, 5),  # Small variations
                "anomaly_score": random.uniform(0.0, 0.2),  # Similar to previous
                "context": baseline_motion + random.uniform(-0.05, 0.05),  # Minimal motion
                "urgency": random.uniform(0.0, 0.1)  # Low priority
            }
            labels.append(0)

        samples.append(sample)

    anomaly_rate = sum(labels) / len(labels)
    print(f"Actual scene change rate: {anomaly_rate:.1%}")
    return samples, labels

def generate_uniform_data(n_samples: int) -> Tuple[List[Dict], List[int]]:
    """Generate uniform data (worst case for adaptive algorithms)"""
    print(f"Generating uniform dataset: {n_samples} samples (all similar significance)")

    samples = []
    labels = []

    # All samples have similar significance - should be challenging for adaptation
    base_significance = 0.5

    for i in range(n_samples):
        sample = {
            "magnitude": base_significance * 100 + random.uniform(-2, 2),
            "anomaly_score": base_significance + random.uniform(-0.02, 0.02),
            "context": base_significance + random.uniform(-0.02, 0.02),
            "urgency": base_significance + random.uniform(-0.02, 0.02)
        }
        samples.append(sample)
        labels.append(0)  # No true anomalies in uniform data

    return samples, labels

def generate_extreme_distribution(n_samples: int, anomaly_rate: float) -> Tuple[List[Dict], List[int]]:
    """Generate dataset with extreme distributions"""
    print(f"Generating extreme dataset: {n_samples} samples, {anomaly_rate*100:.1f}% anomalies")

    samples = []
    labels = []

    for i in range(n_samples):
        is_anomaly = random.random() < anomaly_rate

        if is_anomaly:
            # Extreme high significance
            sample = {
                "magnitude": 100.0,  # Maximum
                "anomaly_score": 1.0,  # Maximum
                "context": 1.0,  # Maximum
                "urgency": 1.0  # Maximum
            }
            labels.append(1)
        else:
            # Extreme low significance
            sample = {
                "magnitude": 0.0,  # Minimum
                "anomaly_score": 0.0,  # Minimum
                "context": 0.0,  # Minimum
                "urgency": 0.0  # Minimum
            }
            labels.append(0)

    return samples, labels

def generate_burst_anomalies(n_samples: int) -> Tuple[List[Dict], List[int]]:
    """Generate dataset with clustered anomalies (challenging for rate control)"""
    print(f"Generating burst anomaly dataset: {n_samples} samples with clustered events")

    samples = []
    labels = []

    # Create bursts of anomalies followed by quiet periods
    in_burst = False
    burst_remaining = 0
    quiet_remaining = 0

    for i in range(n_samples):
        if burst_remaining > 0:
            # In anomaly burst
            sample = {
                "magnitude": random.uniform(70, 100),
                "anomaly_score": random.uniform(0.7, 1.0),
                "context": random.uniform(0.6, 1.0),
                "urgency": random.uniform(0.8, 1.0)
            }
            labels.append(1)
            burst_remaining -= 1
        elif quiet_remaining > 0:
            # In quiet period
            sample = {
                "magnitude": random.uniform(5, 25),
                "anomaly_score": random.uniform(0.0, 0.2),
                "context": random.uniform(0.0, 0.3),
                "urgency": random.uniform(0.0, 0.1)
            }
            labels.append(0)
            quiet_remaining -= 1
        else:
            # Decide next phase
            if random.random() < 0.1:  # 10% chance to start burst
                burst_remaining = random.randint(10, 30)  # Burst of 10-30 anomalies
                sample = {
                    "magnitude": random.uniform(70, 100),
                    "anomaly_score": random.uniform(0.7, 1.0),
                    "context": random.uniform(0.6, 1.0),
                    "urgency": random.uniform(0.8, 1.0)
                }
                labels.append(1)
                burst_remaining -= 1
            else:
                quiet_remaining = random.randint(50, 200)  # Quiet period
                sample = {
                    "magnitude": random.uniform(5, 25),
                    "anomaly_score": random.uniform(0.0, 0.2),
                    "context": random.uniform(0.0, 0.3),
                    "urgency": random.uniform(0.0, 0.1)
                }
                labels.append(0)
                quiet_remaining -= 1

        samples.append(sample)

    anomaly_rate = sum(labels) / len(labels)
    print(f"Actual burst anomaly rate: {anomaly_rate:.1%}")
    return samples, labels

def test_algorithm_stress(dataset_name: str, samples: List[Dict],
                         ground_truth: List[int], target_rates: List[float]) -> List[Dict]:
    """Stress test algorithm on challenging dataset with multiple target rates"""

    print(f"\nSTRESS TESTING: {dataset_name}")
    print("=" * 60)
    print(f"Dataset size: {len(samples)} samples")
    print(f"True anomaly rate: {sum(ground_truth)/len(ground_truth):.1%}")

    results = []

    for target_rate in target_rates:
        print(f"\n--- Testing {target_rate*100:.0f}% target rate ---")

        # Create fresh algorithm instance
        config = SundewConfig()
        config.target_activation_rate = target_rate
        algorithm = SimpleSundewAlgorithm(config)

        # Track detailed metrics
        activations = []
        threshold_history = []
        detected_anomalies = 0
        total_anomalies = sum(ground_truth)

        # Process all samples
        for i, (sample, is_anomaly) in enumerate(zip(samples, ground_truth)):
            result = algorithm.process(sample)
            activated = result is not None
            activations.append(activated)

            # Track threshold evolution
            report = algorithm.report()
            threshold_history.append(report['threshold'])

            # Track anomaly detection
            if activated and is_anomaly:
                detected_anomalies += 1

        # Calculate comprehensive metrics
        final_report = algorithm.report()

        # Rate control metrics
        achieved_rate = final_report['activation_rate']
        rate_error = abs(achieved_rate - target_rate)
        energy_savings = final_report['energy_savings_pct']

        # Stability metrics
        threshold_volatility = np.std(threshold_history[-100:]) if len(threshold_history) >= 100 else (np.std(threshold_history) if threshold_history else 0)
        threshold_range = (max(threshold_history) - min(threshold_history)) if threshold_history else 0

        # Detection metrics
        recall = detected_anomalies / total_anomalies if total_anomalies > 0 else 0
        activated_normal = sum(1 for act, gt in zip(activations, ground_truth) if act and gt == 0)
        precision = detected_anomalies / (detected_anomalies + activated_normal) if (detected_anomalies + activated_normal) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        result = {
            'dataset': dataset_name,
            'target_rate': target_rate,
            'achieved_rate': achieved_rate,
            'rate_error': rate_error,
            'energy_savings': energy_savings,
            'recall': recall,
            'precision': precision,
            'f1_score': f1_score,
            'threshold_volatility': threshold_volatility,
            'threshold_range': threshold_range,
            'final_threshold': final_report['threshold'],
            'total_anomalies': total_anomalies,
            'detected_anomalies': detected_anomalies
        }

        # Success criteria (more lenient for difficult datasets)
        rate_ok = rate_error < 0.08  # Allow 8% error for difficult data
        energy_ok = energy_savings > 50  # Allow lower energy savings
        stability_ok = threshold_volatility < 0.2  # Check for oscillation

        print(f"Target: {target_rate:.1%} -> Achieved: {achieved_rate:.1%} (error: {rate_error:.1%})")
        print(f"Energy saved: {energy_savings:.1f}%, Recall: {recall:.1%}, F1: {f1_score:.3f}")
        print(f"Threshold volatility: {threshold_volatility:.3f}, Range: {threshold_range:.3f}")

        status = "PASS" if (rate_ok and energy_ok and stability_ok) else "STRUGGLE"
        print(f"Status: {status}")

        results.append(result)

    return results

def main():
    """Run comprehensive stress tests before git commit"""

    print("SUNDEW ALGORITHM - COMPREHENSIVE STRESS TEST")
    print("=" * 60)
    print("Testing algorithm robustness before git push")
    print("Includes video processing, adversarial cases, and edge conditions")
    print()

    all_results = []

    # Test 1: Video Frame Processing
    print("TEST 1: VIDEO FRAME PROCESSING")
    print("-" * 40)
    video_samples, video_labels = generate_video_frame_dataset(1000, scene_change_rate=0.02)
    video_results = test_algorithm_stress("Video Frames", video_samples, video_labels, [0.1, 0.2, 0.3])
    all_results.extend(video_results)

    # Test 2: Uniform Data (Adversarial)
    print("\nTEST 2: UNIFORM DATA (ADVERSARIAL)")
    print("-" * 40)
    uniform_samples, uniform_labels = generate_uniform_data(1000)
    uniform_results = test_algorithm_stress("Uniform Data", uniform_samples, uniform_labels, [0.15, 0.25, 0.35])
    all_results.extend(uniform_results)

    # Test 3: Extreme Distributions
    print("\nTEST 3: EXTREME DISTRIBUTIONS")
    print("-" * 40)
    extreme_samples, extreme_labels = generate_extreme_distribution(1000, anomaly_rate=0.01)
    extreme_results = test_algorithm_stress("Extreme Distribution", extreme_samples, extreme_labels, [0.1, 0.2])
    all_results.extend(extreme_results)

    # Test 4: Burst Anomalies
    print("\nTEST 4: BURST ANOMALIES")
    print("-" * 40)
    burst_samples, burst_labels = generate_burst_anomalies(1000)
    burst_results = test_algorithm_stress("Burst Anomalies", burst_samples, burst_labels, [0.15, 0.25])
    all_results.extend(burst_results)

    # Comprehensive Analysis
    print("\n" + "=" * 60)
    print("COMPREHENSIVE STRESS TEST RESULTS")
    print("=" * 60)
    print("Dataset                  Target   Achieved  Error    Energy   Recall   F1     Volatility")
    print("-" * 85)

    passes = 0
    for result in all_results:
        rate_ok = result['rate_error'] < 0.08
        energy_ok = result['energy_savings'] > 50
        stability_ok = result['threshold_volatility'] < 0.2

        status = "PASS" if (rate_ok and energy_ok and stability_ok) else "STRUGGLE"
        if status == "PASS":
            passes += 1

        print(f"{result['dataset']:22s} {result['target_rate']:7.1%}  "
              f"{result['achieved_rate']:8.1%}  {result['rate_error']:6.1%}  "
              f"{result['energy_savings']:7.1f}%  {result['recall']:7.1%}  "
              f"{result['f1_score']:6.3f}  {result['threshold_volatility']:8.3f}")

    # Overall Assessment
    total_tests = len(all_results)
    pass_rate = passes / total_tests

    print(f"\nOVERALL STRESS TEST RESULTS: {passes}/{total_tests} tests passed ({pass_rate:.1%})")

    if pass_rate >= 0.8:
        print("\nEXCELLENT! Algorithm is robust across challenging scenarios.")
        print("* Handles video processing workloads")
        print("* Stable on uniform/adversarial data")
        print("* Adapts to extreme distributions")
        print("* Manages burst anomaly patterns")
        print("\nREADY FOR GIT PUSH!")
    elif pass_rate >= 0.6:
        print(f"\nGOOD! Algorithm handles most challenging scenarios.")
        print(f"Some edge cases may need monitoring in production.")
        print(f"PROCEED WITH CAUTION for git push.")
    else:
        print(f"\nCONCERN! Algorithm struggles with {(1-pass_rate)*100:.0f}% of challenging scenarios.")
        print(f"Consider additional robustness improvements before git push.")

    # Detailed Analysis
    video_tests = [r for r in all_results if "Video" in r['dataset']]
    if video_tests:
        avg_video_energy = sum(r['energy_savings'] for r in video_tests) / len(video_tests)
        avg_video_recall = sum(r['recall'] for r in video_tests) / len(video_tests)
        print(f"\nVIDEO PROCESSING PERFORMANCE:")
        print(f"Average energy savings: {avg_video_energy:.1f}%")
        print(f"Average scene change detection: {avg_video_recall:.1%}")

    uniform_tests = [r for r in all_results if "Uniform" in r['dataset']]
    if uniform_tests:
        avg_uniform_volatility = sum(r['threshold_volatility'] for r in uniform_tests) / len(uniform_tests)
        print(f"\nUNIFORM DATA STABILITY:")
        print(f"Average threshold volatility: {avg_uniform_volatility:.3f} (lower is better)")

    return pass_rate >= 0.6

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
