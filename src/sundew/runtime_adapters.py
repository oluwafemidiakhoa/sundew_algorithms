# src/sundew/runtime_adapters.py
"""Adapters bridging the legacy ``SundewAlgorithm`` into the pipeline runtime."""
from __future__ import annotations
import random

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .gating import gate_probability_with_hysteresis
from .interfaces import (
    ControlPolicy,
    ControlState,
    EnergyModel,
    GatingDecision,
    GatingStrategy,
    ProcessingContext,
    SignificanceModel,
)

if TYPE_CHECKING:  # pragma: no cover - avoid circular import at runtime
    from .core import SundewAlgorithm


class LegacySignificanceModel(SignificanceModel):
    """Adapter wrapping ``SundewAlgorithm._compute_significance``."""

    def __init__(self, algo: "SundewAlgorithm") -> None:
        self.algo = algo

    def compute_significance(
        self,
        context: ProcessingContext,
    ) -> Tuple[float, Dict[str, float]]:
        significance = float(self.algo._compute_significance(context.features))
        return significance, {
            "w_magnitude": float(self.algo.cfg.w_magnitude),
            "w_anomaly": float(self.algo.cfg.w_anomaly),
            "w_context": float(self.algo.cfg.w_context),
            "w_urgency": float(self.algo.cfg.w_urgency),
        }

    def update(self, context: ProcessingContext, outcome: Optional[Dict[str, Any]]) -> None:
        return None

    def get_parameters(self) -> Dict[str, float]:
        return {
            "w_magnitude": float(self.algo.cfg.w_magnitude),
            "w_anomaly": float(self.algo.cfg.w_anomaly),
            "w_context": float(self.algo.cfg.w_context),
            "w_urgency": float(self.algo.cfg.w_urgency),
        }

    def set_parameters(self, params: Dict[str, float]) -> None:
        for key in ("w_magnitude", "w_anomaly", "w_context", "w_urgency"):
            if key in params:
                setattr(self.algo.cfg, key, float(params[key]))


class LegacyControlPolicy(ControlPolicy):
    """Wrap the existing ``SundewAlgorithm._adapt_threshold`` logic."""

    def __init__(self, algo: "SundewAlgorithm") -> None:
        self.algo = algo

    def update_threshold(
        self,
        current_state: ControlState,
        target_activation_rate: float,
        recent_activations: List[bool],
        energy_state: Dict[str, float],
    ) -> Tuple[float, ControlState]:
        activated: Optional[bool]
        if recent_activations:
            activated = bool(recent_activations[-1])
        else:
            activated = None

        if getattr(self.algo, "_threshold_already_updated", False):
            setattr(self.algo, "_threshold_already_updated", False)
        else:
            self.algo._adapt_threshold(activated=activated)

        new_threshold = float(getattr(self.algo, "threshold", current_state.threshold))
        energy_level = float(getattr(self.algo.energy, "value", energy_state.get("level", 0.0)))

        updated_state = ControlState(
            threshold=new_threshold,
            activation_rate=float(self.algo.metrics.ema_activation_rate),
            energy_level=energy_level,
            error_integral=float(getattr(self.algo, "_int_err", 0.0)),
            stability_metrics={},
        )
        return new_threshold, updated_state

    def predict_stability(self, current_state: ControlState, horizon: int = 100) -> Dict[str, float]:
        return {}

    def get_theoretical_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {}


class LegacyGatingStrategy(GatingStrategy):
    """Adapter for probe cadence, refractory handling, and hysteresis gating."""

    def __init__(self, algo: "SundewAlgorithm") -> None:
        self.algo = algo
        seed = int(getattr(algo.cfg, "rng_seed", 42))
        self._rng = random.Random(seed)

    def gate(
        self,
        significance: float,
        threshold: float,
        context: ProcessingContext,
        control_state: ControlState,
    ) -> GatingDecision:
        algo = self.algo

        processed = max(1, getattr(algo.metrics, "processed", 0))
        eff_probe_every = max(1, getattr(algo, "_eff_probe_every", 1))
        force_probe = processed == 1 or (
            getattr(algo, "_eff_probe_every", 0) > 0
            and processed % eff_probe_every == 0
        )

        if not force_probe and getattr(algo, "_refractory_left", 0) > 0:
            algo._refractory_left -= 1
            algo._tick_dormant_energy()
            algo._adapt_threshold(activated=False)
            setattr(algo, "_threshold_already_updated", True)
            setattr(algo, "_runtime_energy_handled", True)
            algo._last_activation = False
            return GatingDecision(
                should_process=False,
                confidence=0.0,
                significance=significance,
                reasoning={"refractory": True, "force_probe": False, "gate_probability": 0.0},
            )

        if force_probe:
            algo._last_activation = True
            return GatingDecision(
                should_process=True,
                confidence=1.0,
                significance=significance,
                reasoning={"refractory": False, "force_probe": True, "gate_probability": 1.0},
            )

        temperature = max(getattr(algo, "_temp", 0.0), 1e-9)
        gate_prob = gate_probability_with_hysteresis(
            significance,
            threshold,
            temperature,
            getattr(algo, "_last_activation", False),
            getattr(algo, "_hysteresis_gap", 0.0),
        )

        activated = self._rng.random() < gate_prob
        algo._last_activation = activated

        return GatingDecision(
            should_process=activated,
            confidence=gate_prob if activated else 1.0 - gate_prob,
            significance=significance,
            reasoning={
                "force_probe": False,
                "refractory": False,
                "gate_probability": gate_prob,
            },
        )

    def get_exploration_probability(self, control_state: ControlState) -> float:
        return 0.0


class LegacyEnergyModel(EnergyModel):
    """Adapter for ``EnergyAccount`` usage within the legacy algorithm."""

    def __init__(self, algo: "SundewAlgorithm") -> None:
        self.algo = algo

    def compute_processing_cost(
        self,
        significance: float,
        processing_type: str,
        context: ProcessingContext,
    ) -> float:
        return float(
            self.algo._eval_cost + self.algo._base_cost * (0.8 + 0.4 * significance)
        )

    def compute_idle_cost(self, duration: float) -> float:
        return float(self.algo._dorm_cost * max(0.0, duration))

    def update_energy_state(
        self,
        current_energy: float,
        cost: float,
        regeneration: float = 0.0,
    ) -> float:
        if getattr(self.algo, "_runtime_energy_handled", False):
            setattr(self.algo, "_runtime_energy_handled", False)
            return float(getattr(self.algo.energy, "value", current_energy))

        if cost <= 0.0:
            return float(getattr(self.algo.energy, "value", current_energy))

        dormant_cost = getattr(self.algo, "_dorm_cost", 0.0)
        if 0.0 < cost <= dormant_cost + 1e-9:
            self.algo._tick_dormant_energy()
            return float(getattr(self.algo.energy, "value", current_energy))

        self.algo._spend_energy(cost)
        return float(getattr(self.algo.energy, "value", current_energy))

    def predict_energy_trajectory(
        self,
        current_energy: float,
        predicted_activations: List[float],
        horizon: int,
    ) -> List[float]:
        return []

    def get_energy_pressure(self, current_energy: float, max_energy: float) -> float:
        return 0.0
