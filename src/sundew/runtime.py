# src/sundew/runtime.py
"""Unified runtime scaffolding for Sundew algorithms.

This module introduces a pipeline-based runtime that can compose significance,
control, gating, and energy stages. It starts with a simple preset mirroring
``SimpleSundewAlgorithm`` while paving the way for richer configurations.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

from .config import SundewConfig
from .interfaces import (
    ControlPolicy,
    ControlState,
    EnergyModel,
    GatingDecision,
    GatingStrategy,
    ProcessingContext,
    ProcessingResult,
    SignificanceModel,
)
from .runtime_adapters import (
    LegacyControlPolicy,
    LegacyEnergyModel,
    LegacyGatingStrategy,
    LegacySignificanceModel,
)

if TYPE_CHECKING:
    from .core import SundewAlgorithm


# ---------------------------------------------------------------------------
# Metrics containers
# ---------------------------------------------------------------------------


@dataclass
class RuntimeMetrics:
    """Track aggregate statistics for the pipeline runtime."""

    processed: int = 0
    activated: int = 0
    total_processing_time: float = 0.0
    activation_history: List[bool] = field(default_factory=list)
    significance_history: List[float] = field(default_factory=list)
    threshold_history: List[float] = field(default_factory=list)
    energy_spent_processing: float = 0.0
    energy_spent_idle: float = 0.0

    def record_activation(self, activated: bool, window: int = 1024) -> None:
        self.activation_history.append(activated)
        if len(self.activation_history) > window:
            self.activation_history = self.activation_history[-window:]

    def record_threshold(self, threshold: float, window: int = 1024) -> None:
        self.threshold_history.append(threshold)
        if len(self.threshold_history) > window:
            self.threshold_history = self.threshold_history[-window:]

    def record_significance(self, significance: float, window: int = 1024) -> None:
        self.significance_history.append(significance)
        if len(self.significance_history) > window:
            self.significance_history = self.significance_history[-window:]


# ---------------------------------------------------------------------------
# Simple stage implementations (mirroring SimpleSundewAlgorithm behaviour)
# ---------------------------------------------------------------------------


class SimpleSignificanceModel(SignificanceModel):
    """Significance computation using weighted linear blend."""

    def __init__(self, config: SundewConfig) -> None:
        self.w_mag = float(config.w_magnitude)
        self.w_ano = float(config.w_anomaly)
        self.w_ctx = float(config.w_context)
        self.w_urg = float(config.w_urgency)

    def compute_significance(self, context: ProcessingContext) -> Tuple[float, Dict[str, Any]]:
        features = context.features

        def _safe_get(key: str, default: float = 0.0) -> float:
            value = features.get(key, default)
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        magnitude = _safe_get("magnitude", 0.0) / 100.0
        anomaly = _safe_get("anomaly_score", 0.0)
        context_val = _safe_get("context", _safe_get("context_relevance", 0.0))
        urgency = _safe_get("urgency", 0.0)

        magnitude = max(0.0, min(1.0, magnitude))
        anomaly = max(0.0, min(1.0, anomaly))
        context_val = max(0.0, min(1.0, context_val))
        urgency = max(0.0, min(1.0, urgency))

        significance = (
            self.w_mag * magnitude
            + self.w_ano * anomaly
            + self.w_ctx * context_val
            + self.w_urg * urgency
        )
        significance = max(0.0, min(1.0, significance))

        explanation = {
            "model": "simple",
            "feature_contributions": {
                "magnitude": self.w_mag * magnitude,
                "anomaly": self.w_ano * anomaly,
                "context": self.w_ctx * context_val,
                "urgency": self.w_urg * urgency,
            },
        }
        return float(significance), explanation

    def update(self, context: ProcessingContext, outcome: Optional[Dict[str, Any]]) -> None:
        return None

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "weights": {
                "magnitude": self.w_mag,
                "anomaly": self.w_ano,
                "context": self.w_ctx,
                "urgency": self.w_urg,
            }
        }

    def set_parameters(self, params: Dict[str, Any]) -> None:
        weights = params.get("weights", {})
        self.w_mag = float(weights.get("magnitude", self.w_mag))
        self.w_ano = float(weights.get("anomaly", self.w_ano))
        self.w_ctx = float(weights.get("context", self.w_ctx))
        self.w_urg = float(weights.get("urgency", self.w_urg))


class SimpleGatingStrategy(GatingStrategy):
    """Hysteresis-based gating matching the simple algorithm behaviour."""

    def __init__(self, hysteresis_gap: float) -> None:
        self.hysteresis_gap = float(max(0.0, hysteresis_gap))
        self._was_active = False

    def gate(
        self,
        significance: float,
        threshold: float,
        context: ProcessingContext,
        control_state: ControlState,
    ) -> GatingDecision:
        if self._was_active:
            effective_threshold = threshold - self.hysteresis_gap
        else:
            effective_threshold = threshold + self.hysteresis_gap

        should_process = significance > effective_threshold
        self._was_active = should_process

        confidence = abs(significance - effective_threshold)
        confidence = max(0.0, min(1.0, confidence))

        reasoning = {
            "effective_threshold": effective_threshold,
            "hysteresis_gap": self.hysteresis_gap,
        }

        return GatingDecision(
            should_process=should_process,
            confidence=confidence,
            significance=significance,
            reasoning=reasoning,
        )

    def get_exploration_probability(self, control_state: ControlState) -> float:
        return 0.0

    @property
    def was_active(self) -> bool:
        return self._was_active


class SimpleControlPolicy(ControlPolicy):
    """PI controller aligned with ``SimpleSundewAlgorithm`` adaptation rules."""

    def __init__(self, config: SundewConfig, window_size: int = 50) -> None:
        self.kp = float(config.adapt_kp)
        self.ki = float(config.adapt_ki)
        self.integral_clamp = float(config.integral_clamp)
        self.min_threshold = float(config.min_threshold)
        self.max_threshold = float(config.max_threshold)
        self.window_size = max(1, int(window_size))
        self._error_integral = 0.0

    def update_threshold(
        self,
        current_state: ControlState,
        target_activation_rate: float,
        recent_activations: List[bool],
        energy_state: Dict[str, float],
    ) -> Tuple[float, ControlState]:
        if len(recent_activations) < 10:
            return current_state.threshold, current_state

        window = min(self.window_size, len(recent_activations))
        window_slice = recent_activations[-window:]
        current_rate = sum(1 for flag in window_slice if flag) / float(window)

        error = target_activation_rate - current_rate
        self._error_integral += error
        self._error_integral = max(
            -self.integral_clamp,
            min(self.integral_clamp, self._error_integral),
        )

        adjustment = self.kp * error + self.ki * self._error_integral
        new_threshold = current_state.threshold - adjustment
        new_threshold = max(self.min_threshold, min(self.max_threshold, new_threshold))

        updated_state = ControlState(
            threshold=new_threshold,
            activation_rate=current_rate,
            energy_level=energy_state.get("level", current_state.energy_level),
            error_integral=self._error_integral,
            stability_metrics={"recent_activation_rate": current_rate},
        )
        return new_threshold, updated_state

    def predict_stability(self, current_state: ControlState, horizon: int = 100) -> Dict[str, float]:
        return {"predicted_activation_rate": current_state.activation_rate}

    def get_theoretical_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            "threshold": (self.min_threshold, self.max_threshold),
            "integral": (-self.integral_clamp, self.integral_clamp),
        }

    @property
    def error_integral(self) -> float:
        return self._error_integral


class SimpleEnergyModel(EnergyModel):
    """Minimal energy accounting used by the simple pipeline preset."""

    def __init__(self, processing_cost: float = 1.0, idle_cost: float = 0.0) -> None:
        self.processing_cost = float(processing_cost)
        self.idle_cost = float(idle_cost)

    def compute_processing_cost(
        self, significance: float, processing_type: str, context: ProcessingContext
    ) -> float:
        return self.processing_cost

    def compute_idle_cost(self, duration: float) -> float:
        return self.idle_cost * max(0.0, duration)

    def update_energy_state(
        self, current_energy: float, cost: float, regeneration: float = 0.0
    ) -> float:
        return max(0.0, current_energy - cost + regeneration)

    def predict_energy_trajectory(
        self, current_energy: float, predicted_activations: Sequence[float], horizon: int
    ) -> List[float]:
        return [float(current_energy)] * max(1, horizon)

    def get_energy_pressure(self, current_energy: float, max_energy: float) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# Pipeline runtime
# ---------------------------------------------------------------------------


class PipelineRuntime:
    """Composable runtime coordinating Sundew stages."""

    def __init__(
        self,
        config: SundewConfig,
        significance_model: SignificanceModel,
        gating_strategy: GatingStrategy,
        control_policy: ControlPolicy,
        energy_model: EnergyModel,
        history_limit: int = 200,
    ) -> None:
        self.cfg = config
        self.significance_model = significance_model
        self.gating_strategy = gating_strategy
        self.control_policy = control_policy
        self.energy_model = energy_model
        self.metrics = RuntimeMetrics()
        self._listeners: List[Callable[[ProcessingResult, Dict[str, Any]], None]] = []
        self._probe_activations = 0

        self._threshold = float(config.activation_threshold)
        self._target_rate = float(config.target_activation_rate)
        self._history_limit = max(1, history_limit)
        self._context_history: List[Dict[str, Any]] = []
        self._sequence_id = 0
        self._energy_level = float(config.max_energy)
        self._legacy_passthrough: bool = False

        # Legacy integration support
        # Backwards compatibility: some significance models carry an embedded
        # legacy algorithm instance.
        self._legacy_algo: Optional["SundewAlgorithm"] = getattr(
            significance_model, "algo", None
        )
        if self._legacy_algo is not None:
            self._energy_level = float(getattr(self._legacy_algo.energy, "value", self._energy_level))

    @property
    def threshold(self) -> float:
        return self._threshold

    def process(self, features: Dict[str, Any]) -> ProcessingResult:
        self.metrics.processed += 1
        self._sequence_id += 1

        legacy_algo = self._legacy_algo
        if legacy_algo is not None:
            legacy_algo.metrics.processed += 1

        context_history = self._context_history[-self._history_limit :]
        processing_context = ProcessingContext(
            timestamp=time.time(),
            sequence_id=self._sequence_id,
            features=features,
            history=context_history,
            metadata={},
        )

        significance, explanation = self.significance_model.compute_significance(
            processing_context
        )
        self.metrics.record_significance(significance)

        current_rate = self._current_activation_rate()
        control_state = ControlState(
            threshold=self._threshold,
            activation_rate=current_rate,
            energy_level=self._energy_level,
            error_integral=getattr(self.control_policy, "error_integral", 0.0),
            stability_metrics={},
        )

        decision = self.gating_strategy.gate(
            significance,
            self._threshold,
            processing_context,
            control_state,
        )

        activated = bool(decision.should_process)
        gate_prob = float(decision.reasoning.get("gate_probability", 1.0 if activated else 0.0))
        refractory_skip = bool(decision.reasoning.get("refractory", False))

        self.metrics.record_activation(activated)
        self.metrics.record_threshold(self._threshold)

        processing_time = (0.001 + 0.001 * (1.0 + significance)) if activated else 0.0
        if activated:
            self.metrics.activated += 1
            if features.get("probe_hint"):
                self._probe_activations += 1

        if legacy_algo is not None and not refractory_skip:
            current_energy = float(getattr(legacy_algo.energy, "value", self._energy_level))
            legacy_algo._collect_telemetry(significance, activated, gate_prob, current_energy, features)

        if activated:
            energy_cost = self.energy_model.compute_processing_cost(
                significance, "inference", processing_context
            )
            self.metrics.energy_spent_processing += energy_cost
        else:
            energy_cost = self.energy_model.compute_idle_cost(1.0)
            self.metrics.energy_spent_idle += energy_cost

        self._energy_level = self.energy_model.update_energy_state(
            self._energy_level,
            energy_cost,
            regeneration=0.0,
        )

        if legacy_algo is not None:
            self._energy_level = float(getattr(legacy_algo.energy, "value", self._energy_level))
            if not refractory_skip:
                legacy_algo._adapt_threshold(activated=activated)
                setattr(legacy_algo, "_threshold_already_updated", True)
            if activated:
                legacy_algo.metrics.activated += 1
                legacy_algo.metrics.total_processing_time += processing_time

        self.metrics.total_processing_time += processing_time

        new_threshold, _ = self.control_policy.update_threshold(
            control_state,
            self._target_rate,
            self.metrics.activation_history,
            {"level": self._energy_level},
        )
        self._threshold = new_threshold

        if legacy_algo is not None:
            legacy_algo.threshold = new_threshold

        self._context_history.append(features)
        if len(self._context_history) > self._history_limit:
            self._context_history = self._context_history[-self._history_limit :]

        component_metrics = {
            "control": {
                "threshold": self._threshold,
                "target_activation_rate": self._target_rate,
            },
            "gating": {
                "confidence": decision.confidence,
                "gate_probability": gate_prob,
            },
            "energy": {
                "level": self._energy_level,
                "cost": energy_cost,
            },
        }

        result = ProcessingResult(
            activated=activated,
            significance=significance,
            energy_consumed=energy_cost,
            processing_time=processing_time,
            threshold_used=self._threshold,
            explanation=explanation,
            component_metrics=component_metrics,
        )

        for listener in self._listeners:
            try:
                listener(result, component_metrics)
            except Exception:
                continue

        return result

    def add_listener(
        self, callback: Callable[[ProcessingResult, Dict[str, Any]], None]
    ) -> None:
        """Register a callback invoked after each processed event."""
        self._listeners.append(callback)

    def report(self) -> Dict[str, Any]:
        processed = max(1, self.metrics.processed)
        activation_rate = self.metrics.activated / float(processed)
        avg_processing_time = (
            self.metrics.total_processing_time / self.metrics.activated
            if self.metrics.activated > 0
            else 0.0
        )

        baseline_cost = processed * float(getattr(self.energy_model, "processing_cost", 1.0))
        actual_cost = self.metrics.energy_spent_processing + self.metrics.energy_spent_idle
        if baseline_cost <= 0:
            energy_savings_pct = 0.0
        else:
            energy_savings_pct = max(0.0, (1.0 - actual_cost / baseline_cost) * 100.0)

        return {
            "samples_processed": self.metrics.processed,
            "samples_activated": self.metrics.activated,
            "probe_activations": self._probe_activations,
            "activation_rate": activation_rate,
            "energy_savings_pct": energy_savings_pct,
            "threshold": self._threshold,
            "target_rate": self._target_rate,
            "avg_processing_time": avg_processing_time,
            "total_processing_time": self.metrics.total_processing_time,
        }

    def _current_activation_rate(self) -> float:
        if not self.metrics.activation_history:
            return 0.0
        window = min(50, len(self.metrics.activation_history))
        recent = self.metrics.activation_history[-window:]
        return sum(1 for flag in recent if flag) / float(window)




class LegacyRuntimeAdapter:
    """Passthrough runtime that delegates directly to ``SundewAlgorithm``."""

    def __init__(self, algo: "SundewAlgorithm") -> None:
        self.algo = algo
        self.metrics = RuntimeMetrics()
        self._context_history: List[Dict[str, Any]] = []
        self._sequence_id = 0

    @property
    def threshold(self) -> float:
        return float(getattr(self.algo, "threshold", 0.0))

    def process(self, features: Dict[str, Any]) -> ProcessingResult:
        prev_energy = float(getattr(self.algo.energy, "value", 0.0))
        significance = float(self.algo._compute_significance(features))

        result = self.algo._process_legacy(features)
        activated = result is not None

        self._sequence_id += 1
        self._context_history.append(features)
        if len(self._context_history) > 1000:
            self._context_history = self._context_history[-1000:]

        self.metrics.processed = self.algo.metrics.processed
        self.metrics.activated = self.algo.metrics.activated
        self.metrics.total_processing_time = self.algo.metrics.total_processing_time
        self.metrics.energy_spent_processing = self.algo.metrics.energy_spent_on_processing
        self.metrics.energy_spent_idle = self.algo.metrics.energy_spent_on_dormancy
        self.metrics.record_significance(significance)
        self.metrics.record_activation(activated)
        self.metrics.record_threshold(self.threshold)

        energy_after = float(getattr(self.algo.energy, "value", prev_energy))
        if result is not None:
            energy_consumed = float(result.energy_consumed)
            processing_time = float(result.processing_time)
            significance_out = float(result.significance)
        else:
            energy_consumed = max(0.0, prev_energy - energy_after)
            processing_time = 0.0
            significance_out = significance

        return ProcessingResult(
            activated=activated,
            significance=significance_out,
            energy_consumed=energy_consumed,
            processing_time=processing_time,
            threshold_used=self.threshold,
            explanation={},
            component_metrics={"legacy": {"passthrough": 1.0 if activated else 0.0}},
        )

    def report(self) -> Dict[str, Any]:
        return self.algo.report()


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_simple_runtime(config: SundewConfig) -> PipelineRuntime:
    """Construct a pipeline runtime that mirrors ``SimpleSundewAlgorithm``."""
    significance = SimpleSignificanceModel(config)
    gating = SimpleGatingStrategy(config.hysteresis_gap)
    control = SimpleControlPolicy(config, window_size=50)
    energy = SimpleEnergyModel()
    return PipelineRuntime(
        config=config,
        significance_model=significance,
        gating_strategy=gating,
        control_policy=control,
        energy_model=energy,
        history_limit=200,
    )


def build_legacy_runtime(algo: "SundewAlgorithm") -> LegacyRuntimeAdapter:
    """Construct a runtime wired to an existing ``SundewAlgorithm`` instance."""
    return LegacyRuntimeAdapter(algo)
