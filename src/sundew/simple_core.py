# src/sundew/simple_core.py
"""
Simplified Sundew algorithm implemented as a compatibility layer around the
new pipeline runtime. The public surface mirrors the previous lightweight
class so existing tests and integrations continue to work while benefiting
from the shared infrastructure.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import SundewConfig
from .interfaces import ProcessingContext, ProcessingResult
from .runtime import PipelineRuntime, build_simple_runtime


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
    """Compatibility wrapper that delegates to ``PipelineRuntime``."""

    def __init__(self, config: SundewConfig) -> None:
        self.cfg = config
        self._runtime: PipelineRuntime = build_simple_runtime(config)

        # Maintain legacy-facing attributes
        self.target_rate = float(config.target_activation_rate)
        self.hysteresis_gap = float(config.hysteresis_gap)
        self.metrics = SimpleMetrics()
        self.error_integral = 0.0
        self.kp = float(config.adapt_kp)
        self.ki = float(config.adapt_ki)
        self.max_integral = float(config.integral_clamp)
        self.min_threshold = float(config.min_threshold)
        self.max_threshold = float(config.max_threshold)
        self.w_mag = float(config.w_magnitude)
        self.w_anom = float(config.w_anomaly)
        self.w_ctx = float(config.w_context)
        self.w_urg = float(config.w_urgency)
        self.was_active = False
        self.activation_history: List[int] = []
        self.window_size = 50

    # ------------------------------------------------------------------
    # Properties mirroring legacy behaviour
    # ------------------------------------------------------------------
    @property
    def threshold(self) -> float:
        return self._runtime.threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        # Allow manual adjustment for callers/tests emulating legacy behaviour
        self._runtime._threshold = float(value)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _update_simple_metrics(self) -> None:
        runtime_metrics = self._runtime.metrics
        self.metrics.processed = runtime_metrics.processed
        self.metrics.activated = runtime_metrics.activated
        self.metrics.total_processing_time = runtime_metrics.total_processing_time

    def _record_activation(self, activated: bool) -> None:
        self.activation_history.append(1 if activated else 0)
        max_len = self.window_size * 2
        if len(self.activation_history) > max_len:
            self.activation_history = self.activation_history[-self.window_size :]

    def _make_context(self, features: Dict[str, Any]) -> ProcessingContext:
        return ProcessingContext(
            timestamp=time.time(),
            sequence_id=self._runtime._sequence_id + 1,
            features=features,
            history=[],
            metadata={},
        )

    # ------------------------------------------------------------------
    # Legacy API surface
    # ------------------------------------------------------------------
    def _compute_significance(self, x: Dict[str, Any]) -> float:
        context = self._make_context(x)
        significance, _ = self._runtime.significance_model.compute_significance(context)
        return float(significance)

    def process(self, x: Dict[str, Any]) -> Optional[SimpleProcessingResult]:
        result: ProcessingResult = self._runtime.process(x)

        self._update_simple_metrics()
        self._record_activation(result.activated)
        self.error_integral = getattr(self._runtime.control_policy, 'error_integral', self.error_integral)
        self.was_active = getattr(self._runtime.gating_strategy, 'was_active', result.activated)

        if not result.activated:
            return None

        # Successful activation mirrors legacy result object
        return SimpleProcessingResult(
            significance=result.significance,
            processing_time=result.processing_time,
        )

    def report(self) -> Dict[str, Any]:
        return self._runtime.report()
