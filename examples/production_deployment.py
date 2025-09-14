#!/usr/bin/env python3
"""
Production Deployment Example

Demonstrates how to deploy Sundew algorithms in production environments with:
- Edge device optimization
- Real-time monitoring
- Robust error handling
- Performance profiling
- Auto-scaling based on load

Usage:
    python examples/production_deployment.py --platform [edge|cloud|hybrid]
"""

import argparse
import json
import logging
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Dict, Generator, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from sundew.enhanced_core import EnhancedSundewAlgorithm, EnhancedSundewConfig
from sundew.monitoring import RealTimeMonitor, MonitoringConfig
from sundew.energy_models import RealisticEnergyConfig


class ProductionConfig:
    """Production deployment configuration."""
    
    def __init__(self, platform: str = "edge"):
        self.platform = platform
        
        # Platform-specific configurations
        if platform == "edge":
            self.algorithm_config = self._get_edge_config()
            self.monitoring_config = self._get_edge_monitoring_config()
            self.performance_targets = self._get_edge_targets()
        elif platform == "cloud":
            self.algorithm_config = self._get_cloud_config()
            self.monitoring_config = self._get_cloud_monitoring_config()
            self.performance_targets = self._get_cloud_targets()
        else:  # hybrid
            self.algorithm_config = self._get_hybrid_config()
            self.monitoring_config = self._get_hybrid_monitoring_config()
            self.performance_targets = self._get_hybrid_targets()
    
    def _get_edge_config(self) -> EnhancedSundewConfig:
        """Optimized configuration for edge devices."""
        return EnhancedSundewConfig(
            significance_model="linear",  # Faster inference
            gating_strategy="temperature",
            control_policy="pi",  # Lower computational overhead
            energy_model="realistic",
            enable_online_learning=False,  # Disable for stability
            component_configs={
                "energy_model": {
                    "platform": "cortex_m4",
                    "battery_capacity_mah": 1000,
                    "thermal_throttling_threshold": 65.0,
                    "dvfs_enabled": True
                },
                "control_policy": {
                    "kp": 0.06,  # Conservative tuning
                    "ki": 0.015,
                    "adaptive_gains": False
                }
            }
        )
    
    def _get_cloud_config(self) -> EnhancedSundewConfig:
        """Optimized configuration for cloud deployment."""
        return EnhancedSundewConfig(
            significance_model="neural",
            gating_strategy="adaptive",
            control_policy="mpc",
            energy_model="simple",  # Less important in cloud
            enable_online_learning=True,
            component_configs={
                "significance_model": {
                    "hidden_sizes": [64, 32, 16],
                    "use_temporal_attention": True,
                    "learning_rate": 0.001
                },
                "control_policy": {
                    "prediction_horizon": 25,
                    "control_horizon": 15
                }
            }
        )
    
    def _get_hybrid_config(self) -> EnhancedSundewConfig:
        """Balanced configuration for hybrid deployment."""
        return EnhancedSundewConfig(
            significance_model="neural",
            gating_strategy="adaptive",
            control_policy="pi",  # MPC too heavy for hybrid
            energy_model="realistic",
            enable_online_learning=True,
            component_configs={
                "significance_model": {
                    "hidden_sizes": [32, 16],  # Smaller network
                    "use_temporal_attention": True,
                    "learning_rate": 0.0005
                },
                "energy_model": {
                    "platform": "cortex_a7",  # Mid-range processor
                    "battery_capacity_mah": 2000
                }
            }
        )
    
    def _get_edge_monitoring_config(self) -> MonitoringConfig:
        """Lightweight monitoring for edge devices."""
        return MonitoringConfig(
            buffer_size=1000,  # Limited memory
            sampling_interval=1.0,  # Less frequent sampling
            enable_alerts=True,
            alert_thresholds={
                "energy_level": (0.15, 1.0),  # Critical energy monitoring
                "activation_rate": (0.05, 0.25)
            },
            enable_live_plots=False,  # No GUI on edge
            log_to_file=True,
            enable_profiling=True
        )
    
    def _get_cloud_monitoring_config(self) -> MonitoringConfig:
        """Full-featured monitoring for cloud deployment."""
        return MonitoringConfig(
            buffer_size=50000,
            sampling_interval=0.1,
            enable_alerts=True,
            alert_thresholds={
                "activation_rate": (0.05, 0.30),
                "f1_score": (0.60, 1.0),
                "oscillation_index": (0.0, 0.10)
            },
            enable_live_plots=True,
            log_to_file=True,
            enable_profiling=True,
            profile_cpu_usage=True,
            profile_memory_usage=True
        )
    
    def _get_hybrid_monitoring_config(self) -> MonitoringConfig:
        """Balanced monitoring for hybrid deployment."""
        return MonitoringConfig(
            buffer_size=10000,
            sampling_interval=0.5,
            enable_alerts=True,
            enable_live_plots=False,
            log_to_file=True,
            enable_profiling=True
        )
    
    def _get_edge_targets(self) -> Dict[str, float]:
        """Performance targets for edge deployment."""
        return {
            "max_latency_ms": 10.0,
            "min_battery_life_hours": 24.0,
            "max_memory_mb": 50.0,
            "min_f1_score": 0.70,
            "max_thermal_throttling": 0.05
        }
    
    def _get_cloud_targets(self) -> Dict[str, float]:
        """Performance targets for cloud deployment."""
        return {
            "max_latency_ms": 50.0,
            "min_throughput_samples_sec": 1000.0,
            "max_memory_mb": 500.0,
            "min_f1_score": 0.85,
            "max_cpu_usage": 0.80
        }
    
    def _get_hybrid_targets(self) -> Dict[str, float]:
        """Performance targets for hybrid deployment."""
        return {
            "max_latency_ms": 25.0,
            "min_battery_life_hours": 12.0,
            "max_memory_mb": 200.0,
            "min_f1_score": 0.75,
            "max_thermal_throttling": 0.10
        }


class DataStreamSimulator:
    """Simulates real-time data streams for testing."""
    
    def __init__(self, stream_type: str = "sensor", rate_hz: float = 10.0):
        self.stream_type = stream_type
        self.rate_hz = rate_hz
        self.running = False
        self.sample_count = 0
        
    def start_stream(self) -> Generator[Dict[str, Any], None, None]:
        """Generate continuous data stream."""
        self.running = True
        self.sample_count = 0
        
        while self.running:
            sample = self._generate_sample()
            yield sample
            time.sleep(1.0 / self.rate_hz)
    
    def stop_stream(self):
        """Stop the data stream."""
        self.running = False
    
    def _generate_sample(self) -> Dict[str, Any]:
        """Generate a single data sample."""
        self.sample_count += 1
        
        if self.stream_type == "sensor":
            # IoT sensor data
            return {
                "magnitude": 45 + 30 * np.sin(self.sample_count * 0.1) + 5 * np.random.normal(),
                "anomaly_score": max(0, min(1, 0.3 + 0.2 * np.random.normal())),
                "context_relevance": 0.5 + 0.3 * (self.sample_count % 20) / 20,
                "urgency": 0.2 + 0.3 * (1 if self.sample_count % 50 < 10 else 0),
                "timestamp": time.time(),
                "source": "sensor_01"
            }
        elif self.stream_type == "video":
            # Video frame analysis
            activity_burst = (self.sample_count % 200) < 30  # Periodic activity
            
            return {
                "magnitude": 60 + (20 if activity_burst else -10) + 8 * np.random.normal(),
                "anomaly_score": (0.7 if activity_burst else 0.2) + 0.1 * np.random.normal(),
                "context_relevance": 0.8 if activity_burst else 0.3,
                "urgency": 0.6 if activity_burst else 0.1,
                "timestamp": time.time(),
                "source": "camera_feed"
            }
        else:  # audio
            # Audio event detection
            event_probability = 0.15
            is_event = np.random.random() < event_probability
            
            return {
                "magnitude": (75 if is_event else 35) + 10 * np.random.normal(),
                "anomaly_score": (0.8 if is_event else 0.2) + 0.1 * np.random.normal(),
                "context_relevance": 0.7 if is_event else 0.4,
                "urgency": 0.8 if is_event else 0.2,
                "timestamp": time.time(),
                "source": "microphone_array"
            }


class ProductionDeployment:
    """Main production deployment manager."""
    
    def __init__(self, platform: str = "edge"):
        self.platform = platform
        self.config = ProductionConfig(platform)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.algorithm = EnhancedSundewAlgorithm(self.config.algorithm_config)
        self.monitor = RealTimeMonitor(self.config.monitoring_config)
        
        # Setup alert handling
        self._setup_alert_handling()
        
        # Performance tracking
        self.performance_metrics = {
            "samples_processed": 0,
            "average_latency": 0.0,
            "current_throughput": 0.0,
            "error_count": 0,
            "uptime_start": time.time()
        }
        
        # Threading for concurrent processing
        self.processing_queue = Queue(maxsize=1000)
        self.result_queue = Queue()
        self.shutdown_event = threading.Event()
        
        self.logger.info(f"Production deployment initialized for {platform} platform")
    
    def _setup_logging(self):
        """Setup production logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'sundew_production_{self.platform}.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('SundewProduction')
    
    def _setup_alert_handling(self):
        """Setup production alert handling."""
        
        def production_alert_handler(alert_type: str, alert_data: Dict[str, Any]):
            severity = alert_data.get("severity", "LOW")
            metric = alert_data.get("metric", "unknown")
            value = alert_data.get("value", 0.0)
            
            if severity in ["HIGH", "CRITICAL"]:
                self.logger.error(f"CRITICAL ALERT: {metric} = {value:.3f}")
                
                # Take corrective action for critical alerts
                if metric == "energy_level" and value < 0.2:
                    self._handle_low_energy()
                elif metric == "oscillation_index" and value > 0.15:
                    self._handle_instability()
                elif metric == "f1_score" and value < 0.5:
                    self._handle_poor_performance()
            else:
                self.logger.warning(f"Alert: {metric} = {value:.3f}")
        
        self.monitor.alert_manager.register_alert_callback(production_alert_handler)
    
    def _handle_low_energy(self):
        """Handle low energy situations."""
        self.logger.info("Handling low energy - switching to conservative mode")
        
        # Temporarily increase threshold to reduce activations
        current_threshold = self.algorithm.control_state.threshold
        self.algorithm.control_state.threshold = min(0.95, current_threshold + 0.1)
        
        # Log the change
        self.logger.info(f"Threshold increased from {current_threshold:.3f} to {self.algorithm.control_state.threshold:.3f}")
    
    def _handle_instability(self):
        """Handle system instability."""
        self.logger.info("Handling system instability - reducing controller gains")
        
        # If using PI controller, reduce gains
        if hasattr(self.algorithm.control_policy, 'current_kp'):
            self.algorithm.control_policy.current_kp *= 0.8
            self.algorithm.control_policy.current_ki *= 0.8
            self.logger.info("Reduced PI controller gains")
    
    def _handle_poor_performance(self):
        """Handle poor performance."""
        self.logger.info("Handling poor performance - analyzing recent patterns")
        
        # Could trigger model retraining, parameter adjustment, etc.
        # For now, just log the issue
        recent_metrics = self.monitor.get_monitoring_summary()
        self.logger.info(f"Recent performance: {recent_metrics['current_metrics']}")
    
    def start_processing(self, data_stream: DataStreamSimulator):
        """Start production processing with real-time monitoring."""
        
        self.logger.info("Starting production processing...")
        self.monitor.start_monitoring(self.algorithm)
        
        # Start worker threads
        processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        metrics_thread = threading.Thread(target=self._metrics_worker, daemon=True)
        
        processing_thread.start()
        metrics_thread.start()
        
        try:
            # Main processing loop
            start_time = time.time()
            
            for sample in data_stream.start_stream():
                if self.shutdown_event.is_set():
                    break
                
                # Add sample to processing queue
                try:
                    self.processing_queue.put(sample, timeout=0.1)
                except:
                    self.logger.warning("Processing queue full, dropping sample")
                    continue
                
                # Performance tracking
                self.performance_metrics["samples_processed"] += 1
                
                # Periodic performance reporting
                if self.performance_metrics["samples_processed"] % 1000 == 0:
                    self._report_performance()
                
                # Check performance targets
                if self.performance_metrics["samples_processed"] % 100 == 0:
                    self._check_performance_targets()
        
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        except Exception as e:
            self.logger.error(f"Processing error: {e}")
        finally:
            self._shutdown()
    
    def _processing_worker(self):
        """Worker thread for processing samples."""
        
        while not self.shutdown_event.is_set():
            try:
                # Get sample from queue
                sample = self.processing_queue.get(timeout=1.0)
                
                # Process with timing
                start_time = time.perf_counter()
                
                try:
                    result = self.algorithm.process(sample)
                    processing_time = time.perf_counter() - start_time
                    
                    # Update performance metrics
                    self._update_latency_metric(processing_time * 1000)  # Convert to ms
                    
                    # Store result
                    result_data = {
                        "sample": sample,
                        "result": result,
                        "processing_time": processing_time,
                        "timestamp": time.time()
                    }
                    self.result_queue.put(result_data)
                    
                    # Update monitoring
                    prediction = 1 if result else 0
                    # In production, would have ground truth from external source
                    ground_truth = self._estimate_ground_truth(sample)
                    self.monitor.update(self.algorithm, ground_truth, prediction)
                
                except Exception as e:
                    self.logger.error(f"Sample processing error: {e}")
                    self.performance_metrics["error_count"] += 1
                
                self.processing_queue.task_done()
                
            except Empty:
                continue  # Timeout, check shutdown event
            except Exception as e:
                self.logger.error(f"Processing worker error: {e}")
    
    def _metrics_worker(self):
        """Worker thread for metrics collection and reporting."""
        
        while not self.shutdown_event.is_set():
            try:
                # Process results and update metrics
                try:
                    result_data = self.result_queue.get(timeout=1.0)
                    
                    # Update throughput
                    current_time = time.time()
                    self.performance_metrics["current_throughput"] = (
                        self.performance_metrics["samples_processed"] / 
                        (current_time - self.performance_metrics["uptime_start"])
                    )
                    
                    self.result_queue.task_done()
                    
                except Empty:
                    continue
                
                time.sleep(0.1)  # Throttle metrics updates
                
            except Exception as e:
                self.logger.error(f"Metrics worker error: {e}")
    
    def _estimate_ground_truth(self, sample: Dict[str, Any]) -> Optional[int]:
        """Estimate ground truth for monitoring (placeholder)."""
        # In production, this would come from external labels/feedback
        # For demo, use a simple heuristic
        
        magnitude = sample.get("magnitude", 0)
        anomaly = sample.get("anomaly_score", 0)
        
        if magnitude > 70 and anomaly > 0.6:
            return 1
        elif magnitude < 40 and anomaly < 0.3:
            return 0
        else:
            return None  # Uncertain cases
    
    def _update_latency_metric(self, latency_ms: float):
        """Update running average latency."""
        alpha = 0.1  # EMA smoothing factor
        self.performance_metrics["average_latency"] = (
            alpha * latency_ms + 
            (1 - alpha) * self.performance_metrics["average_latency"]
        )
    
    def _report_performance(self):
        """Report current performance metrics."""
        uptime = time.time() - self.performance_metrics["uptime_start"]
        
        self.logger.info(f"Performance Report:")
        self.logger.info(f"  Samples Processed: {self.performance_metrics['samples_processed']}")
        self.logger.info(f"  Average Latency: {self.performance_metrics['average_latency']:.2f} ms")
        self.logger.info(f"  Throughput: {self.performance_metrics['current_throughput']:.1f} samples/sec")
        self.logger.info(f"  Error Rate: {self.performance_metrics['error_count'] / max(1, self.performance_metrics['samples_processed']):.4f}")
        self.logger.info(f"  Uptime: {uptime:.1f} seconds")
        
        # Algorithm-specific metrics
        algo_report = self.algorithm.get_comprehensive_report()
        self.logger.info(f"  Activation Rate: {algo_report['activation_rate']:.3f}")
        self.logger.info(f"  Energy Remaining: {algo_report['energy_remaining']:.1f}")
        self.logger.info(f"  Research Quality Score: {algo_report['research_quality_score']:.1f}/10")
    
    def _check_performance_targets(self):
        """Check if performance targets are being met."""
        targets = self.config.performance_targets
        
        # Check latency target
        if "max_latency_ms" in targets:
            if self.performance_metrics["average_latency"] > targets["max_latency_ms"]:
                self.logger.warning(f"Latency target missed: {self.performance_metrics['average_latency']:.2f} > {targets['max_latency_ms']}")
        
        # Check throughput target (for cloud)
        if "min_throughput_samples_sec" in targets:
            if self.performance_metrics["current_throughput"] < targets["min_throughput_samples_sec"]:
                self.logger.warning(f"Throughput target missed: {self.performance_metrics['current_throughput']:.1f} < {targets['min_throughput_samples_sec']}")
        
        # Check algorithm performance
        algo_report = self.algorithm.get_comprehensive_report()
        
        if "min_f1_score" in targets:
            # Would need recent F1 score from monitoring
            monitoring_summary = self.monitor.get_monitoring_summary()
            current_f1 = monitoring_summary.get("current_metrics", {}).get("f1_score", 0)
            
            if current_f1 > 0 and current_f1 < targets["min_f1_score"]:
                self.logger.warning(f"F1 score target missed: {current_f1:.3f} < {targets['min_f1_score']}")
    
    def _shutdown(self):
        """Graceful shutdown."""
        self.logger.info("Shutting down production deployment...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        # Export final monitoring data
        timestamp = int(time.time())
        self.monitor.export_data(f"production_monitoring_{self.platform}_{timestamp}.json")
        
        # Final performance report
        self._report_performance()
        
        # Export algorithm state
        final_report = self.algorithm.get_comprehensive_report()
        with open(f"production_final_report_{self.platform}_{timestamp}.json", 'w') as f:
            json.dump(final_report, f, indent=2)
        
        self.logger.info("Production deployment shutdown complete")


@contextmanager
def production_deployment_context(platform: str):
    """Context manager for production deployment."""
    deployment = ProductionDeployment(platform)
    try:
        yield deployment
    finally:
        deployment._shutdown()


def main():
    """Main production deployment runner."""
    
    parser = argparse.ArgumentParser(description="Sundew Production Deployment")
    parser.add_argument("--platform", choices=["edge", "cloud", "hybrid"], 
                       default="edge", help="Deployment platform")
    parser.add_argument("--stream-type", choices=["sensor", "video", "audio"],
                       default="sensor", help="Data stream type")
    parser.add_argument("--duration", type=int, default=300,
                       help="Deployment duration in seconds")
    parser.add_argument("--rate", type=float, default=10.0,
                       help="Data stream rate in Hz")
    
    args = parser.parse_args()
    
    print(f"🚀 Sundew Production Deployment - {args.platform.upper()} Platform")
    print("=" * 60)
    
    try:
        # Setup data stream
        data_stream = DataStreamSimulator(args.stream_type, args.rate)
        
        # Deploy with context manager for proper cleanup
        with production_deployment_context(args.platform) as deployment:
            
            print(f"Platform: {args.platform}")
            print(f"Stream Type: {args.stream_type}")
            print(f"Data Rate: {args.rate} Hz")
            print(f"Duration: {args.duration} seconds")
            print(f"Performance Targets: {deployment.config.performance_targets}")
            
            # Start timer for duration limit
            start_time = time.time()
            
            # Override data stream to stop after duration
            original_start_stream = data_stream.start_stream
            
            def limited_stream():
                for sample in original_start_stream():
                    if time.time() - start_time > args.duration:
                        break
                    yield sample
            
            data_stream.start_stream = limited_stream
            
            # Start production processing
            deployment.start_processing(data_stream)
            
        print("\n✅ Production deployment completed successfully!")
        
    except KeyboardInterrupt:
        print("\n🛑 Deployment stopped by user")
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())