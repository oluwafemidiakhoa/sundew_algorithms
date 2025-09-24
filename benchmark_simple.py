#!/usr/bin/env python3
"""
Simple, reproducible benchmark for Sundew algorithm
Tests on synthetic data with known ground truth
"""

import random
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple
import json

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    dataset: str
    target_rate: float
    actual_rate: float
    energy_saved: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    processing_time: float
    threshold_final: float

    @property
    def precision(self) -> float:
        if self.true_positives + self.false_positives == 0:
            return 0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        if self.true_positives + self.false_negatives == 0:
            return 0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1_score(self) -> float:
        if self.precision + self.recall == 0:
            return 0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

def generate_synthetic_stream(n_samples: int, anomaly_rate: float = 0.05) -> Tuple[List[dict], List[bool]]:
    """
    Generate synthetic data stream with known anomalies

    Returns:
        samples: List of sample dictionaries
        labels: List of boolean labels (True = anomaly)
    """
    samples = []
    labels = []

    for i in range(n_samples):
        is_anomaly = random.random() < anomaly_rate

        if is_anomaly:
            # Anomalous sample - high values
            sample = {
                'value': random.uniform(70, 100),
                'variance': random.uniform(15, 30),
                'trend': random.uniform(0.7, 1.0),
                'frequency': random.uniform(0.8, 1.0)
            }
        else:
            # Normal sample - low values
            sample = {
                'value': random.uniform(20, 50),
                'variance': random.uniform(2, 8),
                'trend': random.uniform(0.0, 0.3),
                'frequency': random.uniform(0.1, 0.4)
            }

        samples.append(sample)
        labels.append(is_anomaly)

    return samples, labels

def compute_significance_simple(sample: dict) -> float:
    """Simple significance function for benchmarking"""
    # Normalize and combine features
    sig = 0.4 * (sample['value'] / 100)
    sig += 0.3 * (sample['variance'] / 30)
    sig += 0.2 * sample['trend']
    sig += 0.1 * sample['frequency']
    return min(1.0, max(0.0, sig))

class BenchmarkAlgorithm:
    """Simplified Sundew for benchmarking"""

    def __init__(self, target_rate: float = 0.2):
        self.target_rate = target_rate
        self.threshold = 0.5
        self.activation_history = []
        self.error_sum = 0

    def process(self, sample: dict) -> bool:
        """Process sample and return activation decision"""

        # Compute significance
        sig = compute_significance_simple(sample)

        # Make decision
        activate = sig > self.threshold
        self.activation_history.append(activate)

        # Update threshold every 50 samples
        if len(self.activation_history) % 50 == 0 and len(self.activation_history) >= 50:
            recent_rate = sum(self.activation_history[-50:]) / 50
            error = self.target_rate - recent_rate
            self.error_sum += error

            # PI controller update
            self.threshold += 0.01 * error + 0.001 * self.error_sum
            self.threshold = min(0.95, max(0.05, self.threshold))

        return activate

def run_benchmark(n_samples: int = 10000, target_rate: float = 0.2,
                  anomaly_rate: float = 0.05) -> BenchmarkResult:
    """Run a single benchmark test"""

    print(f"Running benchmark: {n_samples} samples, {target_rate:.1%} target rate, {anomaly_rate:.1%} anomalies")

    # Generate data
    samples, labels = generate_synthetic_stream(n_samples, anomaly_rate)

    # Initialize algorithm
    algo = BenchmarkAlgorithm(target_rate)

    # Process samples
    start_time = time.time()
    activations = []

    for sample in samples:
        activate = algo.process(sample)
        activations.append(activate)

    processing_time = time.time() - start_time

    # Calculate metrics
    tp = sum(1 for a, l in zip(activations, labels) if a and l)
    fp = sum(1 for a, l in zip(activations, labels) if a and not l)
    fn = sum(1 for a, l in zip(activations, labels) if not a and l)
    tn = sum(1 for a, l in zip(activations, labels) if not a and not l)

    actual_rate = sum(activations) / len(activations)
    energy_saved = 1.0 - actual_rate

    return BenchmarkResult(
        dataset="synthetic",
        target_rate=target_rate,
        actual_rate=actual_rate,
        energy_saved=energy_saved,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        processing_time=processing_time,
        threshold_final=algo.threshold
    )

def main():
    """Run comprehensive benchmark suite"""

    print("Sundew Algorithm Benchmark")
    print("=" * 60)
    print()

    results = []

    # Test different target rates
    for target_rate in [0.1, 0.15, 0.2, 0.25, 0.3]:
        result = run_benchmark(
            n_samples=10000,
            target_rate=target_rate,
            anomaly_rate=0.05
        )
        results.append(result)

        print(f"  Target: {target_rate:.1%}, Actual: {result.actual_rate:.1%}")
        print(f"  Energy saved: {result.energy_saved:.1%}")
        print(f"  F1 Score: {result.f1_score:.3f}")
        print(f"  Precision: {result.precision:.3f}, Recall: {result.recall:.3f}")
        print(f"  Processing time: {result.processing_time:.2f}s")
        print()

    # Test different anomaly rates
    print("Testing robustness to different anomaly rates:")
    print("-" * 60)

    for anomaly_rate in [0.01, 0.05, 0.10, 0.20]:
        result = run_benchmark(
            n_samples=5000,
            target_rate=0.2,
            anomaly_rate=anomaly_rate
        )

        print(f"  Anomaly rate: {anomaly_rate:.1%}")
        print(f"  F1 Score: {result.f1_score:.3f}")
        print(f"  Energy saved: {result.energy_saved:.1%}")
        print()

    # Save results
    with open('benchmark_results.json', 'w') as f:
        json.dump([{
            'dataset': r.dataset,
            'target_rate': r.target_rate,
            'actual_rate': r.actual_rate,
            'energy_saved': r.energy_saved,
            'f1_score': r.f1_score,
            'precision': r.precision,
            'recall': r.recall,
            'processing_time': r.processing_time
        } for r in results], f, indent=2)

    print("Results saved to benchmark_results.json")

    # Summary
    print("\nSummary:")
    print("-" * 60)
    avg_energy = np.mean([r.energy_saved for r in results])
    avg_f1 = np.mean([r.f1_score for r in results])
    print(f"Average energy saved: {avg_energy:.1%}")
    print(f"Average F1 score: {avg_f1:.3f}")
    print(f"Rate control accuracy: ±{np.std([abs(r.actual_rate - r.target_rate) for r in results]):.3f}")

if __name__ == "__main__":
    main()
