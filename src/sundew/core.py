# src/sundew/core.py
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .config import SundewConfig
from .energy import EnergyAccount
from .gating import gate_probability


@dataclass(slots=True)
class ProcessingResult:
    """Record returned when an event is processed (activated)."""

    significance: float
    processing_time: float
    energy_consumed: float


@dataclass(slots=True)
class Metrics:
    """Minimal metrics container (queried by demo/tests)."""

    ema_activation_rate: float = 0.0
    processed: int = 0
    activated: int = 0
    total_processing_time: float = 0.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_get(d: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(d.get(key, default))
    except Exception:
        return float(default)


class SundewAlgorithm:
    """
    Sundew reference implementation with PI control, energy pressure, and gating.
    Includes deterministic probe cadence and optional refractory cooldown.
    """

    def __init__(self, config: SundewConfig) -> None:
        config.validate()
        self.cfg = config

        # Threshold/controller state
        self.threshold: float = float(self.cfg.activation_threshold)
        self._int_err: float = 0.0

        # Metrics
        self.metrics: Metrics = Metrics(ema_activation_rate=0.0)

        # Hot-path cache
        self._ema_alpha: float = float(self.cfg.ema_alpha)
        self._kp: float = float(self.cfg.adapt_kp)
        self._ki: float = float(self.cfg.adapt_ki)
        self._dead: float = float(self.cfg.error_deadband)
        self._min_thr: float = float(self.cfg.min_threshold)
        self._max_thr: float = float(self.cfg.max_threshold)
        self._press: float = float(self.cfg.energy_pressure)
        self._temp: float = float(self.cfg.gate_temperature)

        self._eval_cost: float = float(self.cfg.eval_cost)
        self._base_cost: float = float(self.cfg.base_processing_cost)
        self._dorm_cost: float = float(self.cfg.dormant_tick_cost)
        self._regen_min, self._regen_max = self.cfg.dormancy_regen

        # Optional extras
        self._probe_every_cfg: int = int(getattr(self.cfg, "probe_every", 0) or 0)
        self._refractory_cfg: int = int(getattr(self.cfg, "refractory", 0) or 0)
        self._refractory_left: int = 0

        # Effective probe cadence (never 0). Default to 100 if unset.
        self._eff_probe_every: int = max(1, (self._probe_every_cfg or 100))

        # Energy account
        max_e = float(self.cfg.max_energy)
        self.energy: EnergyAccount = EnergyAccount(max_e, max_e)

        # RNG (for probabilistic gating)
        random.seed(int(self.cfg.rng_seed))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(self, x: Dict[str, Any]) -> Optional[ProcessingResult]:
        self.metrics.processed += 1

        # Deterministic probe
        force_probe = self.metrics.processed == 1 or (
            self._eff_probe_every > 0 and (self.metrics.processed % self._eff_probe_every == 0)
        )

        # Respect refractory only when not forcing a probe
        if not force_probe and self._refractory_left > 0:
            self._refractory_left -= 1
            self._tick_dormant_energy()
            self._adapt_threshold(activated=False)
            return None

        sig = self._compute_significance(x)

        # Gate decision
        if force_probe:
            activated = True
        else:
            if self._temp <= 1e-9:
                activated = sig >= self.threshold
            else:
                p = gate_probability(sig, self.threshold, max(self._temp, 1e-9))
                activated = random.random() < p

        if not activated:
            self._tick_dormant_energy()
            self._adapt_threshold(activated=False)
            return None

        # Activated
        start = time.perf_counter()
        proc_time = 0.001 + 0.001 * (1.0 + sig)  # ~1–2 ms
        _ = start + proc_time  # shape only; no sleep

        energy_used = self._eval_cost + self._base_cost * (0.8 + 0.4 * sig)
        self._spend_energy(energy_used)

        self.metrics.activated += 1
        self.metrics.total_processing_time += proc_time

        if self._refractory_cfg > 0:
            self._refractory_left = self._refractory_cfg

        self._adapt_threshold(activated=True)

        return ProcessingResult(
            significance=float(sig),
            processing_time=float(proc_time),
            energy_consumed=float(energy_used),
        )

    def report(self) -> Dict[str, Any]:
        n = max(1, self.metrics.processed)
        act_rate = self.metrics.activated / n
        avg_pt = (
            (self.metrics.total_processing_time / self.metrics.activated)
            if self.metrics.activated
            else 0.0
        )

        energy_remaining = float(getattr(self.energy, "value", 0.0))

        baseline_energy_cost = n * (self._eval_cost + self._base_cost)
        actual_energy_cost = (
            self.metrics.activated * (self._eval_cost + self._base_cost)
            + (n - self.metrics.activated) * self._dorm_cost
        )
        savings_pct = (
            (1.0 - (actual_energy_cost / baseline_energy_cost)) * 100.0
            if baseline_energy_cost > 0
            else 0.0
        )

        return {
            "total_inputs": int(self.metrics.processed),
            "activations": int(self.metrics.activated),
            "activation_rate": float(act_rate),
            "ema_activation_rate": float(self.metrics.ema_activation_rate),
            "avg_processing_time": float(avg_pt),
            "total_energy_spent": float(self.cfg.max_energy - energy_remaining),
            "energy_remaining": float(energy_remaining),
            "threshold": float(self.threshold),
            "baseline_energy_cost": float(baseline_energy_cost),
            "actual_energy_cost": float(actual_energy_cost),
            "estimated_energy_savings_pct": float(savings_pct),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _compute_significance(self, x: Dict[str, Any]) -> float:
        mag = _safe_get(x, "magnitude", 0.0) / 100.0
        ano = _safe_get(x, "anomaly_score", 0.0)
        ctx = _safe_get(x, "context_relevance", 0.0)
        urg = _safe_get(x, "urgency", 0.0)
        s = (
            self.cfg.w_magnitude * mag
            + self.cfg.w_anomaly * ano
            + self.cfg.w_context * ctx
            + self.cfg.w_urgency * urg
        )
        return _clamp(s, 0.0, 1.0)

    def _adapt_threshold(self, activated: Optional[bool] = None) -> None:
        if activated is not None:
            obs = 1.0 if activated else 0.0
            a = self._ema_alpha
            self.metrics.ema_activation_rate = (
                a * obs + (1.0 - a) * self.metrics.ema_activation_rate
            )

        err = float(self.cfg.target_activation_rate) - self.metrics.ema_activation_rate
        if abs(err) <= self._dead:
            err = 0.0

        self._int_err = _clamp(
            self._int_err + err, -self.cfg.integral_clamp, self.cfg.integral_clamp
        )

        delta = self._kp * err + self._ki * self._int_err

        frac = float(getattr(self.energy, "value", 0.0)) / float(self.cfg.max_energy)
        press = self._press * (1.0 - _clamp(frac, 0.0, 1.0))

        self.threshold = _clamp(self.threshold - delta + press, self._min_thr, self._max_thr)

    def _tick_dormant_energy(self) -> None:
        v = float(getattr(self.energy, "value", 0.0))
        v = max(0.0, v - self._dorm_cost)
        v = min(float(self.cfg.max_energy), v + random.uniform(self._regen_min, self._regen_max))
        self.energy.value = v

    def _spend_energy(self, amount: float) -> None:
        v = float(getattr(self.energy, "value", 0.0))
        self.energy.value = max(0.0, v - float(amount))
