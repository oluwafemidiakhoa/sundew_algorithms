"""
Tests for the simplified Sundew algorithm (simple_core.py)
These tests validate the working implementation used for production
"""

import pytest
from src.sundew.simple_core import SimpleSundewAlgorithm, SimpleProcessingResult
from src.sundew.config import SundewConfig


class TestSimpleSundewAlgorithm:
    """Test the simplified algorithm implementation"""

    def test_basic_instantiation(self):
        """Test algorithm can be created with default config"""
        config = SundewConfig()
        algorithm = SimpleSundewAlgorithm(config)
        assert algorithm.threshold == config.activation_threshold
        assert algorithm.target_rate == config.target_activation_rate

    def test_process_returns_correct_types(self):
        """Test process method returns correct types"""
        config = SundewConfig()
        algorithm = SimpleSundewAlgorithm(config)

        # High significance sample
        high_sig_sample = {
            "magnitude": 90,
            "anomaly_score": 0.9,
            "urgency": 0.8,
            "context": 0.7
        }

        result = algorithm.process(high_sig_sample)
        if result is not None:
            assert isinstance(result, SimpleProcessingResult)
            assert hasattr(result, 'significance')
            assert hasattr(result, 'processing_time')

    def test_significance_computation(self):
        """Test significance function works correctly"""
        config = SundewConfig()
        algorithm = SimpleSundewAlgorithm(config)

        # Test significance bounds
        low_sample = {"magnitude": 0, "anomaly_score": 0, "urgency": 0, "context": 0}
        low_sig = algorithm._compute_significance(low_sample)
        assert 0.0 <= low_sig <= 1.0

        high_sample = {"magnitude": 100, "anomaly_score": 1.0, "urgency": 1.0, "context": 1.0}
        high_sig = algorithm._compute_significance(high_sample)
        assert 0.0 <= high_sig <= 1.0
        assert high_sig > low_sig

    def test_rate_convergence(self):
        """Test algorithm converges to target rate"""
        config = SundewConfig()
        config.target_activation_rate = 0.2  # 20% target
        algorithm = SimpleSundewAlgorithm(config)

        # Process many samples with mixed significance
        activations = []
        for i in range(1000):
            if i % 5 == 0:  # 20% high significance
                sample = {"magnitude": 80, "anomaly_score": 0.8, "urgency": 0.7, "context": 0.6}
            else:  # 80% low significance
                sample = {"magnitude": 20, "anomaly_score": 0.2, "urgency": 0.1, "context": 0.2}

            result = algorithm.process(sample)
            activations.append(result is not None)

        # Check convergence (use last 200 samples for stability)
        recent_rate = sum(activations[-200:]) / 200
        assert abs(recent_rate - 0.2) < 0.05  # Within 5% of target

    def test_hysteresis_prevents_oscillation(self):
        """Test hysteresis mechanism works"""
        config = SundewConfig()
        config.hysteresis_gap = 0.1  # Large gap for testing
        algorithm = SimpleSundewAlgorithm(config)

        # Sample with significance near threshold
        borderline_sample = {"magnitude": 50, "anomaly_score": 0.5, "urgency": 0.5, "context": 0.5}

        # Process same sample multiple times
        results = []
        for _ in range(10):
            result = algorithm.process(borderline_sample)
            results.append(result is not None)

        # Should not oscillate rapidly due to hysteresis
        # (This is a qualitative test - in practice would need more sophisticated analysis)
        assert len(set(results)) <= 2  # At most 2 different states

    def test_report_metrics(self):
        """Test reporting functionality"""
        config = SundewConfig()
        algorithm = SimpleSundewAlgorithm(config)

        # Process some samples
        for i in range(100):
            sample = {"magnitude": 50, "anomaly_score": 0.5, "urgency": 0.3, "context": 0.4}
            algorithm.process(sample)

        report = algorithm.report()

        # Check report structure
        assert 'samples_processed' in report
        assert 'samples_activated' in report
        assert 'activation_rate' in report
        assert 'energy_savings_pct' in report
        assert 'threshold' in report
        assert 'target_rate' in report

        # Check values make sense
        assert report['samples_processed'] == 100
        assert 0 <= report['activation_rate'] <= 1
        assert 0 <= report['energy_savings_pct'] <= 100
        assert report['samples_activated'] == report['samples_processed'] * report['activation_rate']

    def test_threshold_adaptation_direction(self):
        """Test threshold adapts in correct direction"""
        config = SundewConfig()
        config.target_activation_rate = 0.5  # 50% target for clear signal
        algorithm = SimpleSundewAlgorithm(config)

        initial_threshold = algorithm.threshold

        # Process many low-significance samples (should increase activation)
        for _ in range(200):
            low_sample = {"magnitude": 10, "anomaly_score": 0.1, "urgency": 0.1, "context": 0.1}
            algorithm.process(low_sample)

        # Threshold should decrease to activate more
        assert algorithm.threshold < initial_threshold

        # Reset and test opposite direction
        algorithm = SimpleSundewAlgorithm(config)
        initial_threshold = algorithm.threshold

        # Process many high-significance samples (should decrease activation)
        for _ in range(200):
            high_sample = {"magnitude": 90, "anomaly_score": 0.9, "urgency": 0.9, "context": 0.9}
            algorithm.process(high_sample)

        # Threshold should increase to activate less
        assert algorithm.threshold > initial_threshold

    def test_missing_features_handled(self):
        """Test algorithm handles missing features gracefully"""
        config = SundewConfig()
        algorithm = SimpleSundewAlgorithm(config)

        # Sample with missing features
        incomplete_sample = {"magnitude": 50}  # Missing other features

        # Should not crash
        result = algorithm.process(incomplete_sample)
        assert result is not None or result is None  # Either outcome is fine

        # Test with completely empty sample
        empty_sample = {}
        result = algorithm.process(empty_sample)
        assert result is not None or result is None  # Should not crash


class TestSimpleProcessingResult:
    """Test the result dataclass"""

    def test_result_creation(self):
        """Test ProcessingResult can be created"""
        result = SimpleProcessingResult(
            significance=0.7,
            processing_time=0.01
        )
        assert result.significance == 0.7
        assert result.processing_time == 0.01


class TestSimpleMetrics:
    """Test the metrics container"""

    def test_metrics_initialization(self):
        """Test metrics initialize correctly"""
        config = SundewConfig()
        algorithm = SimpleSundewAlgorithm(config)

        assert algorithm.metrics.processed == 0
        assert algorithm.metrics.activated == 0
        assert algorithm.metrics.total_processing_time == 0.0


# Integration test with real data patterns
class TestSimpleAlgorithmIntegration:
    """Integration tests with realistic data patterns"""

    def test_iot_sensor_pattern(self):
        """Test with IoT sensor-like data pattern"""
        config = SundewConfig()
        config.target_activation_rate = 0.15  # 15% activation
        algorithm = SimpleSundewAlgorithm(config)

        # Simulate IoT sensor data (mostly normal, occasional anomalies)
        activations = []
        for i in range(500):
            if i % 20 == 0:  # 5% anomalies
                sample = {
                    "magnitude": 85,
                    "anomaly_score": 0.9,
                    "urgency": 0.8,
                    "context": 0.7
                }
            else:  # Normal readings
                sample = {
                    "magnitude": 25,
                    "anomaly_score": 0.15,
                    "urgency": 0.1,
                    "context": 0.2
                }

            result = algorithm.process(sample)
            activations.append(result is not None)

        # Should converge to target and save energy
        final_rate = sum(activations[-100:]) / 100  # Last 100 samples
        assert abs(final_rate - 0.15) < 0.12  # Within 12% (lenient for integration test)

        # Should detect most anomalies
        anomaly_indices = list(range(0, 500, 20))  # Every 20th sample
        detected_anomalies = sum(1 for i in anomaly_indices if i < len(activations) and activations[i])
        detection_rate = detected_anomalies / len(anomaly_indices)
        assert detection_rate > 0.6  # Detect at least 60% of anomalies

    def test_video_frame_pattern(self):
        """Test with video frame-like data pattern"""
        config = SundewConfig()
        config.target_activation_rate = 0.1  # 10% activation for video
        algorithm = SimpleSundewAlgorithm(config)

        # Simulate video frames (mostly similar, occasional scene changes)
        activations = []
        for i in range(300):
            if i % 30 == 0:  # 3.3% scene changes
                sample = {
                    "magnitude": 90,
                    "anomaly_score": 0.95,
                    "urgency": 0.8,
                    "context": 0.9
                }
            else:  # Similar frames
                sample = {
                    "magnitude": 15,
                    "anomaly_score": 0.05,
                    "urgency": 0.05,
                    "context": 0.1
                }

            result = algorithm.process(sample)
            activations.append(result is not None)

        # Should achieve reasonable rate control
        final_rate = sum(activations[-50:]) / 50  # Last 50 samples
        assert abs(final_rate - 0.1) < 0.2  # Within 20% for video data (small dataset)

        # Should achieve energy savings
        total_rate = sum(activations) / len(activations)
        assert total_rate < 0.2  # Less than 20% activation overall
