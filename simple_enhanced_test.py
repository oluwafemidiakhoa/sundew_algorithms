#!/usr/bin/env python3
"""Simple test of enhanced Sundew system with error handling."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_enhanced():
    print("Testing Enhanced Sundew - Basic Components")

    try:
        from sundew.enhanced_core import EnhancedSundewAlgorithm, EnhancedSundewConfig
        print("[OK] Enhanced core imported")

        # Test with minimal config
        config = EnhancedSundewConfig(
            significance_model="linear",
            gating_strategy="temperature",
            control_policy="pi",
            energy_model="simple",
            target_activation_rate=0.15,
            enable_online_learning=False,  # Disable to avoid complexity
            enable_auto_tuning=False
        )
        print("[OK] Config created")

        algorithm = EnhancedSundewAlgorithm(config)
        print("[OK] Algorithm initialized")

        # Test single sample
        test_sample = {
            "magnitude": 50.0,
            "anomaly_score": 0.3,
            "context_relevance": 0.4,
            "urgency": 0.2
        }

        print("Testing single sample...")
        start_time = time.time()
        result = algorithm.process(test_sample)
        processing_time = time.time() - start_time

        print(f"[OK] Processed sample in {processing_time*1000:.2f}ms")
        print(f"  - Activated: {result.activated}")
        print(f"  - Significance: {result.significance:.3f}")
        print(f"  - Energy consumed: {result.energy_consumed:.3f}")

        # Test comprehensive report
        report = algorithm.get_comprehensive_report()
        print("[OK] Got comprehensive report")
        print(f"  - Research quality score: {report['research_quality_score']:.1f}/10")
        print(f"  - Available keys: {list(report.keys())}")
        print(f"  - Total processed: {report.get('total_processed', 'N/A')}")

        return True

    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_enhanced()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
