# src/sundew/demo.py
from __future__ import annotations

import random as _random
import time
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List

from .config import SundewConfig
from .core import ProcessingResult, SundewAlgorithm


# ---------------------------------------------------------------------
# Terminal emoji safety (Windows codepages / CI logs can choke on them)
# ---------------------------------------------------------------------
def _stdout_supports_unicode() -> bool:
    import sys

    enc = getattr(sys.stdout, "encoding", None) or ""
    try:
        "🌿".encode(enc or "utf-8", errors="strict")
        return True
    except Exception:
        return False


EMOJI_OK = _stdout_supports_unicode()
BULLET = "🌿" if EMOJI_OK else "[sundew]"
CHECK = "✅" if EMOJI_OK else "[ok]"
PAUSE = "⏸" if EMOJI_OK else "[idle]"
FLAG_DONE = "🏁" if EMOJI_OK else "[done]"

# ---------------------------------------------------------------------
# Event schema with type-specific anomaly/urgency priors
# ---------------------------------------------------------------------
EVENT_TYPES = [
    {"type": "environmental", "anomaly_bias": (0.0, 0.4), "urgency_bias": (0.1, 0.6)},
    {"type": "security", "anomaly_bias": (0.4, 0.9), "urgency_bias": (0.3, 0.8)},
    {"type": "health_monitor", "anomaly_bias": (0.2, 0.7), "urgency_bias": (0.2, 0.7)},
    {"type": "system_alert", "anomaly_bias": (0.3, 0.8), "urgency_bias": (0.2, 0.9)},
    {"type": "emergency", "anomaly_bias": (0.8, 1.0), "urgency_bias": (0.9, 1.0)},
]

# Simple module-level RNG for reproducibility in CLI/demo usage


_rng = _random.Random(42)


def synth_event(i: int) -> Dict:  # pragma: no cover
    """
    Sample a synthetic event (bounded in [0,1] except magnitude ~[0,100]).
    Kept simple so CLI can call without passing an RNG.
    """
    kind = _rng.choice(EVENT_TYPES)
    mag_raw = _rng.uniform(0.0, 1.0)  # base in [0,1]
    anomaly = _rng.uniform(*kind["anomaly_bias"])
    urgency = _rng.uniform(*kind["urgency_bias"])
    context = _rng.uniform(0.0, 1.0)

    return {
        "id": f"event_{i:05d}",
        "type": kind["type"],
        "magnitude": mag_raw * 100.0,  # core expects ~[0,100]
        "anomaly_score": anomaly,  # [0,1]
        "context_relevance": context,  # [0,1]
        "urgency": urgency,  # [0,1]
        "timestamp": time.time(),
    }


# ---------------------------------------------------------------------
# Public programmatic demo entry (used by CLI/tests)
# ---------------------------------------------------------------------
def run_demo(n_events: int = 40, temperature: float = 0.1) -> Dict:  # pragma: no cover
    """
    Run a small synthetic stream and return a structured report.
    Randomness is seeded deterministically for reproducibility.
    """
    # Configure and seed
    cfg = SundewConfig(gate_temperature=temperature)
    algo = SundewAlgorithm(cfg)

    processed: List[ProcessingResult] = []

    print(f"{BULLET} Sundew Algorithm — Demo")
    print("=" * 60)
    # EnergyAccount exposes .value; fall back to float(algo.energy)
    try:
        energy_val = float(getattr(algo.energy, "value", algo.energy))
    except Exception:
        energy_val = 0.0
    print(f"Initial threshold: {algo.threshold:.3f} | Energy: {energy_val:.1f}\n")

    for i in range(n_events):
        x = synth_event(i)
        res = algo.process(x)

        # re-read energy to reflect changes
        try:
            energy_val = float(getattr(algo.energy, "value", algo.energy))
        except Exception:
            energy_val = 0.0

        if res is None:
            print(
                f"{i + 1:02d}. {x['type']:<15} {PAUSE} dormant "
                f"| energy {energy_val:6.1f} | thr {algo.threshold:.3f}"
            )
        else:
            processed.append(res)
            print(
                f"{i + 1:02d}. {x['type']:<15} {CHECK} processed "
                f"(sig={res.significance:.3f}, {res.processing_time:.3f}s, ΔE≈{res.energy_consumed:.1f}) "
                f"| energy {energy_val:6.1f} | thr {algo.threshold:.3f}"
            )

    print(f"\n{FLAG_DONE} Final Report")
    report = algo.report()
    for k, v in report.items():
        if isinstance(v, float):
            if "pct" in k:
                print(f"  {k:30s}: {v:7.2f}%")
            else:
                print(f"  {k:30s}: {v:10.3f}")
        else:
            print(f"  {k:30s}: {v}")

    # Safer serialization for dataclasses (ProcessingResult is a dataclass)
    proc_serializable = []
    for r in processed:
        try:
            proc_serializable.append(asdict(r))
        except Exception:
            proc_serializable.append(getattr(r, "__dict__", {}))

    return {
        "config": asdict(cfg),
        "report": report,
        "processed_events": proc_serializable,
        "generated_at": datetime.utcnow().isoformat(),
    }
