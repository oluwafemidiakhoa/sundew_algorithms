#!/usr/bin/env python3
"""
Test the fixed Sundew algorithm on REAL datasets with ground truth
This is the crucial test to validate the algorithm actually works
"""

import pandas as pd
import numpy as np
from src.sundew.simple_core import SimpleSundewAlgorithm
from src.sundew.config import SundewConfig
from typing import Dict, List, Tuple

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

def test_algorithm_on_dataset(dataset_name: str, samples: List[Dict],
                            ground_truth: List[int], target_rate: float = 0.2) -> Dict:
    """Test algorithm on real dataset with ground truth"""

    print(f"\nTesting on {dataset_name}")
    print("=" * 60)
    print(f"Dataset size: {len(samples)} samples")
    print(f"Target activation rate: {target_rate:.1%}")
    print(f"True anomaly rate: {sum(ground_truth)/len(ground_truth):.1%}")
    print()

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

        # Progress updates
        if len(samples) >= 5 and (i + 1) % (len(samples) // 5) == 0:
            report = algorithm.report()
            print(f"Progress {i+1:5d}/{len(samples)}: Rate={report['activation_rate']:.1%}, "
                  f"Threshold={report['threshold']:.3f}")

    # Calculate metrics
    final_report = algorithm.report()

    # Detection performance
    true_positives = detected_anomalies
    false_negatives = total_anomalies - detected_anomalies

    # Get activated normal samples (false positives)
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

    print(f"\nFINAL RESULTS:")
    print("-" * 40)
    print(f"Target rate: {target_rate:.1%}")
    print(f"Achieved rate: {final_report['activation_rate']:.1%}")
    print(f"Rate error: {results['rate_error']:.1%}")
    print(f"Energy saved: {final_report['energy_savings_pct']:.1f}%")
    print(f"Final threshold: {final_report['threshold']:.3f}")
    print()
    print(f"ANOMALY DETECTION:")
    print(f"Total anomalies: {total_anomalies}")
    print(f"Detected: {detected_anomalies}")
    print(f"Recall: {recall:.1%}")
    print(f"Precision: {precision:.1%}")
    print(f"F1-score: {f1_score:.3f}")

    # Success criteria
    rate_ok = results['rate_error'] < 0.05  # Within 5%
    energy_ok = results['energy_savings'] > 60  # >60% savings
    detection_ok = recall > 0.7  # >70% recall

    print(f"\nSUCCESS CRITERIA:")
    print(f"Rate convergence (<5% error): {'PASS' if rate_ok else 'FAIL'}")
    print(f"Energy savings (>60%): {'PASS' if energy_ok else 'FAIL'}")
    print(f"Anomaly recall (>70%): {'PASS' if detection_ok else 'FAIL'}")

    success = rate_ok and energy_ok and detection_ok
    print(f"\nOVERALL: {'SUCCESS' if success else 'NEEDS IMPROVEMENT'}")

    return results

def main():
    """Run tests on real datasets"""

    print("TESTING SUNDEW ALGORITHM ON REAL DATASETS")
    print("=" * 60)
    print("This test validates the algorithm on your actual data with ground truth")
    print()

    # Test IoT Sensor Data
    print("Loading IoT sensor monitoring data...")
    iot_samples, iot_labels = load_iot_sensor_data()

    if iot_samples:
        # Test with 20% target rate on first 1000 samples
        iot_result = test_algorithm_on_dataset(
            "IoT Sensor Monitoring",
            iot_samples[:1000],
            iot_labels[:1000],
            target_rate=0.2
        )

        print(f"\n" + "="*60)
        print(f"REAL DATA TEST SUMMARY")
        print(f"="*60)
        print(f"Dataset: {iot_result['dataset']}")
        print(f"Samples tested: {iot_result['samples']}")
        print(f"Rate Error: {iot_result['rate_error']:.1%} (target <5%)")
        print(f"Energy Savings: {iot_result['energy_savings']:.1f}% (target >60%)")
        print(f"Anomaly Recall: {iot_result['recall']:.1%} (target >70%)")
        print(f"F1-Score: {iot_result['f1_score']:.3f}")

        # Overall assessment
        if (iot_result['rate_error'] < 0.05 and
            iot_result['energy_savings'] > 60 and
            iot_result['recall'] > 0.7):
            print(f"\nVERDICT: ALGORITHM WORKS ON REAL DATA!")
            print("The Sundew algorithm successfully:")
            print("- Converges to target activation rate")
            print("- Achieves significant energy savings")
            print("- Maintains good anomaly detection")
        else:
            print(f"\nVERDICT: NEEDS IMPROVEMENT ON REAL DATA")
            print("Algorithm may need domain-specific tuning")
    else:
        print("Could not load any real datasets for testing")

if __name__ == "__main__":
    main()
