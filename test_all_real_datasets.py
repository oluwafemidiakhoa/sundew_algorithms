#!/usr/bin/env python3
"""
Test the fixed Sundew algorithm on ALL real datasets with ground truth
This is the comprehensive validation test
"""

import pandas as pd
import numpy as np
from src.sundew.simple_core import SimpleSundewAlgorithm
from src.sundew.config import SundewConfig
from typing import Dict, List, Tuple
import os

def load_iot_sensor_data() -> Tuple[List[Dict], List[int]]:
    """Load IoT sensor monitoring dataset"""
    try:
        df = pd.read_csv('data/raw/iot_sensor_monitoring.csv')
        print(f"Loaded IoT dataset: {len(df)} samples")

        samples = []
        labels = []

        for _, row in df.iterrows():
            sample = {
                "magnitude": float(row['magnitude']),
                "anomaly_score": float(row['anomaly_score']),
                "context": float(row['context_relevance']),
                "urgency": float(row['urgency'])
            }
            samples.append(sample)
            labels.append(int(row['ground_truth']))

        anomaly_rate = sum(labels) / len(labels)
        print(f"Anomaly rate: {anomaly_rate:.1%}")
        return samples, labels

    except Exception as e:
        print(f"Error loading IoT data: {e}")
        return [], []

def load_network_security_data() -> Tuple[List[Dict], List[int]]:
    """Load network security dataset"""
    try:
        df = pd.read_csv('data/raw/network_security.csv')
        print(f"Loaded Network Security dataset: {len(df)} samples")

        samples = []
        labels = []

        for _, row in df.iterrows():
            sample = {
                "magnitude": float(row['magnitude']),
                "anomaly_score": float(row['anomaly_score']),
                "context": float(row['context_relevance']),
                "urgency": float(row['urgency'])
            }
            samples.append(sample)
            labels.append(int(row['ground_truth']))

        anomaly_rate = sum(labels) / len(labels)
        print(f"Anomaly rate: {anomaly_rate:.1%}")
        return samples, labels

    except Exception as e:
        print(f"Error loading Network Security data: {e}")
        return [], []

def load_financial_data() -> Tuple[List[Dict], List[int]]:
    """Load financial time series dataset"""
    try:
        df = pd.read_csv('data/raw/financial_time_series.csv')
        print(f"Loaded Financial dataset: {len(df)} samples")

        samples = []
        labels = []

        for _, row in df.iterrows():
            sample = {
                "magnitude": float(row['magnitude']),
                "anomaly_score": float(row['anomaly_score']),
                "context": float(row['context_relevance']),
                "urgency": float(row['urgency'])
            }
            samples.append(sample)
            labels.append(int(row['ground_truth']))

        anomaly_rate = sum(labels) / len(labels)
        print(f"Anomaly rate: {anomaly_rate:.1%}")
        return samples, labels

    except Exception as e:
        print(f"Error loading Financial data: {e}")
        return [], []

def test_algorithm_on_dataset(dataset_name: str, samples: List[Dict],
                            ground_truth: List[int], target_rate: float = 0.2) -> Dict:
    """Test algorithm on real dataset with ground truth"""

    print(f"\nTesting on {dataset_name}")
    print("=" * 50)
    print(f"Dataset size: {len(samples)} samples")
    print(f"Target activation rate: {target_rate:.1%}")
    print(f"True anomaly rate: {sum(ground_truth)/len(ground_truth):.1%}")

    # Create algorithm
    config = SundewConfig()
    config.target_activation_rate = target_rate
    algorithm = SimpleSundewAlgorithm(config)

    # Track results
    activations = []
    detected_anomalies = 0
    total_anomalies = sum(ground_truth)

    # Process all samples
    for i, (sample, is_anomaly) in enumerate(zip(samples, ground_truth)):
        result = algorithm.process(sample)
        activated = result is not None
        activations.append(activated)

        # Track anomaly detection
        if activated and is_anomaly:
            detected_anomalies += 1

    # Calculate metrics
    final_report = algorithm.report()

    # Detection performance
    true_positives = detected_anomalies
    false_negatives = total_anomalies - detected_anomalies
    activated_normal = sum(1 for act, gt in zip(activations, ground_truth)
                          if act and gt == 0)
    false_positives = activated_normal

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    results = {
        'dataset': dataset_name,
        'samples': len(samples),
        'target_rate': target_rate,
        'achieved_rate': final_report['activation_rate'],
        'rate_error': abs(final_report['activation_rate'] - target_rate),
        'energy_savings': final_report['energy_savings_pct'],
        'final_threshold': final_report['threshold'],
        'total_anomalies': total_anomalies,
        'detected_anomalies': detected_anomalies,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

    print(f"Results: Rate={final_report['activation_rate']:.1%} (error: {results['rate_error']:.1%}), "
          f"Energy={final_report['energy_savings_pct']:.0f}%, Recall={recall:.1%}, F1={f1_score:.3f}")

    return results

def main():
    """Run comprehensive tests on all real datasets"""

    print("COMPREHENSIVE REAL DATASET VALIDATION")
    print("=" * 60)
    print("Testing Sundew algorithm on ALL your real datasets")
    print()

    all_results = []
    datasets = [
        ("IoT Sensor Monitoring", load_iot_sensor_data),
        ("Network Security", load_network_security_data),
        ("Financial Time Series", load_financial_data)
    ]

    for dataset_name, loader_func in datasets:
        print(f"\n{'-'*60}")
        print(f"Loading {dataset_name}...")

        samples, labels = loader_func()

        if samples:
            # Test with subset for speed (1000 samples max)
            test_size = min(1000, len(samples))
            test_samples = samples[:test_size]
            test_labels = labels[:test_size]

            result = test_algorithm_on_dataset(
                dataset_name,
                test_samples,
                test_labels,
                target_rate=0.2
            )
            all_results.append(result)
        else:
            print(f"Failed to load {dataset_name}")

    # Summary of all tests
    print(f"\n{'='*60}")
    print("COMPREHENSIVE RESULTS SUMMARY")
    print(f"{'='*60}")
    print("Dataset                    Samples  Rate_Err  Energy   Recall   F1     Status")
    print("-" * 75)

    successes = 0
    for result in all_results:
        rate_ok = result['rate_error'] < 0.05
        energy_ok = result['energy_savings'] > 60
        recall_ok = result['recall'] > 0.7
        success = rate_ok and energy_ok and recall_ok

        if success:
            successes += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"{result['dataset']:25s} {result['samples']:7d}  "
              f"{result['rate_error']:7.1%}  {result['energy_savings']:6.0f}%  "
              f"{result['recall']:7.1%}  {result['f1_score']:5.3f}  {status}")

    print("\n" + "="*60)
    print(f"OVERALL VALIDATION RESULTS: {successes}/{len(all_results)} datasets PASSED")

    if successes == len(all_results):
        print("\nEXCELLENT! Algorithm works on ALL real datasets!")
        print("Your Sundew algorithm is ready for publication:")
        print("✓ Converges to target rates on real data")
        print("✓ Achieves 60-80% energy savings")
        print("✓ Maintains >70% anomaly detection")
        print("✓ Works across different domains (IoT, Network, Financial)")
    elif successes >= len(all_results) * 0.75:
        print(f"\nGOOD! Algorithm works on {successes}/{len(all_results)} datasets.")
        print("Minor tuning may improve performance on remaining datasets.")
    elif successes >= len(all_results) * 0.5:
        print(f"\nMODERATE. Algorithm works on {successes}/{len(all_results)} datasets.")
        print("Some domain-specific tuning recommended.")
    else:
        print(f"\nNEEDS WORK. Only {successes}/{len(all_results)} datasets passed.")
        print("Significant improvements needed for real-world deployment.")

    # Specific metrics summary
    if all_results:
        avg_rate_error = sum(r['rate_error'] for r in all_results) / len(all_results)
        avg_energy_savings = sum(r['energy_savings'] for r in all_results) / len(all_results)
        avg_recall = sum(r['recall'] for r in all_results) / len(all_results)
        avg_f1 = sum(r['f1_score'] for r in all_results) / len(all_results)

        print(f"\nAVERAGE PERFORMANCE ACROSS ALL REAL DATASETS:")
        print(f"Rate Error: {avg_rate_error:.1%} (lower is better)")
        print(f"Energy Savings: {avg_energy_savings:.1f}% (higher is better)")
        print(f"Anomaly Recall: {avg_recall:.1%} (higher is better)")
        print(f"F1-Score: {avg_f1:.3f} (higher is better)")

if __name__ == "__main__":
    main()
