#!/usr/bin/env python3
"""
Simple test to isolate recursion issue.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sundew.enhanced_core import EnhancedSundewAlgorithm, EnhancedSundewConfig


def simple_test():
    """Test basic functionality without advanced features."""
    # Simple config without advanced features
    config = EnhancedSundewConfig(
        significance_model="linear",
        gating_strategy="temperature",
        control_policy="pi",
        energy_model="simple",
        enable_information_theoretic_threshold=False,
        enable_batch_processing=False,
        enable_automl=False,
        enable_theoretical_analysis=False
    )

    algorithm = EnhancedSundewAlgorithm(config)

    # Single sample test
    sample = {
        'magnitude': 50.0,
        'anomaly_score': 0.5,
        'context_relevance': 0.6,
        'urgency': 0.4
    }

    try:
        result = algorithm.process(sample)
        print(f"Basic test: SUCCESS - Activated: {result.activated}")
        return True
    except Exception as e:
        print(f"Basic test: FAILED - {e}")
        return False

def test_with_info_theory():
    """Test with only information theory enabled."""
    config = EnhancedSundewConfig(
        significance_model="linear",
        gating_strategy="temperature",
        control_policy="pi",
        energy_model="simple",
        enable_information_theoretic_threshold=True,
        information_threshold_method="mutual_information",
        enable_batch_processing=False,
        enable_automl=False,
        enable_theoretical_analysis=False
    )

    try:
        algorithm = EnhancedSundewAlgorithm(config)

        # Process a few samples
        for i in range(5):
            sample = {
                'magnitude': 50.0 + i * 10,
                'anomaly_score': 0.5,
                'context_relevance': 0.6,
                'urgency': 0.4
            }
            algorithm.process(sample)

        print("Info theory test: SUCCESS")
        return True
    except Exception as e:
        print(f"Info theory test: FAILED - {e}")
        return False

if __name__ == "__main__":
    print("Simple Enhanced Algorithm Test")
    print("=" * 40)

    success1 = simple_test()
    success2 = test_with_info_theory()

    if success1 and success2:
        print("\n[OK] Basic enhanced algorithm functionality works")
    else:
        print("\n[ERROR] Enhanced algorithm has fundamental issues")
