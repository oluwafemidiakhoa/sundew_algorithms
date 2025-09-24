# src/sundew/simple_core.py
"""
Simplified Sundew algorithm that actually works
Removes all the complex features that interfere with basic operation
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import SundewConfig


@dataclass
class SimpleProcessingResult:
    """Result returned when an event is processed."""
    significance: float
    processing_time: float


@dataclass
class SimpleMetrics:
    """Simple metrics container."""
    processed: int = 0
    activated: int = 0
    total_processing_time: float = 0.0


class SimpleSundewAlgorithm:
    """
    Simplified Sundew algorithm that focuses on the core functionality:
    1. Score input significance
    2. Compare to adaptive threshold
    3. Use PI controller to maintain target rate
    4. Apply hysteresis for stability

    Removes all energy modeling and complex features that interfere.
    """

    def __init__(self, config: SundewConfig) -> None:
        self.cfg = config

        # Core state
        self.threshold = float(config.activation_threshold)
        self.target_rate = float(config.target_activation_rate)

        # PI controller state
        self.error_integral = 0.0
        self.kp = float(config.adapt_kp)
        self.ki = float(config.adapt_ki)
        self.max_integral = float(config.integral_clamp)

        # Hysteresis for stability
        self.hysteresis_gap = float(config.hysteresis_gap)
        self.was_active = False

        # Activation history for rate calculation
        self.activation_history: List[int] = []
        self.window_size = 50  # Window for rate calculation

        # Significance weights
        self.w_mag = float(config.w_magnitude)
        self.w_anom = float(config.w_anomaly)
        self.w_ctx = float(config.w_context)
        self.w_urg = float(config.w_urgency)

        # Bounds
        self.min_threshold = float(config.min_threshold)
        self.max_threshold = float(config.max_threshold)

        # Metrics
        self.metrics = SimpleMetrics()

    def _compute_significance(self, x: Dict[str, Any]) -> float:
        """Compute significance score from input features."""
        # Extract and normalize features
        magnitude = self._safe_get(x, "magnitude", 0.0) / 100.0  # Assume max 100
        anomaly = self._safe_get(x, "anomaly_score", 0.0)
        context = self._safe_get(x, "context", 0.0)
        urgency = self._safe_get(x, "urgency", 0.0)

        # Weighted combination
        sig = (self.w_mag * magnitude +
               self.w_anom * anomaly +
               self.w_ctx * context +
               self.w_urg * urgency)

        return max(0.0, min(1.0, sig))

    def _safe_get(self, d: Dict[str, Any], key: str, default: float = 0.0) -> float:
        """Safely extract numeric value from dict."""
        val = d.get(key, default)
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def _should_activate(self, significance: float) -> bool:
        """Decide whether to activate based on significance and threshold."""
        # Apply hysteresis to prevent oscillation
        if self.was_active:
            effective_threshold = self.threshold - self.hysteresis_gap
        else:
            effective_threshold = self.threshold + self.hysteresis_gap

        return significance > effective_threshold

    def _update_threshold(self, activated: bool) -> None:
        """Update threshold using PI controller to maintain target rate."""
        # Add to history
        self.activation_history.append(1 if activated else 0)

        # Keep window size manageable
        if len(self.activation_history) > self.window_size * 2:
            self.activation_history = self.activation_history[-self.window_size:]

        # Need enough samples for meaningful rate calculation
        if len(self.activation_history) < 10:
            return

        # Calculate current activation rate from recent window
        window = min(self.window_size, len(self.activation_history))
        recent_activations = self.activation_history[-window:]
        current_rate = sum(recent_activations) / len(recent_activations)

        # PI controller
        error = self.target_rate - current_rate

        # Update integral term with windup protection
        self.error_integral += error
        self.error_integral = max(-self.max_integral,
                                min(self.max_integral, self.error_integral))

        # Calculate threshold adjustment
        # Note: SUBTRACT error because we want to DECREASE threshold when rate is too low
        adjustment = self.kp * error + self.ki * self.error_integral
        self.threshold -= adjustment

        # Apply bounds
        self.threshold = max(self.min_threshold,
                           min(self.max_threshold, self.threshold))

    def process(self, x: Dict[str, Any]) -> Optional[SimpleProcessingResult]:
        """
        Process an input through the gating algorithm.
        Returns ProcessingResult if activated, None if skipped.
        """
        self.metrics.processed += 1

        # Compute significance
        significance = self._compute_significance(x)

        # Make gating decision
        activated = self._should_activate(significance)

        # Update threshold adaptation
        self._update_threshold(activated)

        # Update state
        self.was_active = activated

        if activated:
            self.metrics.activated += 1

            # Simulate processing time
            processing_time = 0.01  # Simulate 10ms processing
            self.metrics.total_processing_time += processing_time

            return SimpleProcessingResult(
                significance=significance,
                processing_time=processing_time
            )
        else:
            # Not activated - save energy
            return None

    def report(self) -> Dict[str, Any]:
        """Generate metrics report."""
        n = max(1, self.metrics.processed)
        activation_rate = self.metrics.activated / n

        avg_processing_time = (
            self.metrics.total_processing_time / self.metrics.activated
            if self.metrics.activated > 0 else 0.0
        )

        # Calculate energy savings (simple model)
        baseline_cost = n * 1.0  # Assume 1 unit per sample if always processing
        actual_cost = self.metrics.activated * 1.0  # Only pay for activated samples
        energy_savings_pct = (1.0 - actual_cost / baseline_cost) * 100.0

        return {
            "samples_processed": n,
            "samples_activated": self.metrics.activated,
            "activation_rate": activation_rate,
            "energy_savings_pct": energy_savings_pct,
            "threshold": self.threshold,
            "target_rate": self.target_rate,
            "avg_processing_time": avg_processing_time,
            "total_processing_time": self.metrics.total_processing_time,
        }
