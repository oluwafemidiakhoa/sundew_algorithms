#!/usr/bin/env python3
"""Generate adversarial event streams to stress Sundew presets."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sundew import build_simple_runtime
from sundew.config_presets import get_preset

FEATURE_KEYS = ("magnitude", "anomaly_score", "context_relevance", "urgency")


def generate_stream(kind: str, n: int, seed: int = 0) -> List[Dict[str, float]]:
    rng = random.Random(seed)
    stream: List[Dict[str, float]] = []
    for i in range(n):
        if kind == "spikes" and i % 50 == 0:
            stream.append({
                "magnitude": rng.uniform(90, 110),
                "anomaly_score": rng.uniform(0.8, 1.0),
                "context_relevance": rng.uniform(0.7, 1.0),
                "urgency": rng.uniform(0.8, 1.0),
            })
        elif kind == "drift":
            base = 20 + (i / n) * 60
            stream.append({
                "magnitude": base + rng.uniform(-5, 5),
                "anomaly_score": min(1.0, 0.2 + (i / n) * 0.6 + rng.uniform(-0.1, 0.1)),
                "context_relevance": min(1.0, 0.2 + (i / n) * 0.5 + rng.uniform(-0.1, 0.1)),
                "urgency": rng.uniform(0.1, 0.4),
            })
        elif kind == "noise":
            if rng.random() < 0.1:
                stream.append({k: rng.uniform(0, 100) for k in FEATURE_KEYS})
            else:
                stream.append({
                    "magnitude": rng.uniform(0, 60),
                    "anomaly_score": rng.uniform(0.0, 0.4),
                    "context_relevance": rng.uniform(0.0, 0.4),
                    "urgency": rng.uniform(0.0, 0.4),
                })
        else:
            stream.append({
                "magnitude": rng.uniform(10, 70),
                "anomaly_score": rng.uniform(0.1, 0.5),
                "context_relevance": rng.uniform(0.1, 0.5),
                "urgency": rng.uniform(0.1, 0.5),
            })
    return stream


def run_scenario(preset: str, kind: str, n: int, seed: int, outdir: Path) -> Dict[str, float]:
    runtime = build_simple_runtime(get_preset(preset))
    stream = generate_stream(kind, n, seed)
    for event in stream:
        runtime.process(event)
    report = runtime.report()
    report["scenario"] = kind
    report["preset"] = preset
    report["samples_processed"] = n
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / f"{preset}_{kind}_seed{seed}.json"
    log_path.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress test Sundew with adversarial streams")
    parser.add_argument("--preset", default="custom_health_hd82")
    parser.add_argument("--scenario", choices=["spikes", "drift", "noise", "baseline"], default="spikes")
    parser.add_argument("--events", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=Path, default=Path("data/results/adversarial_runs"))
    args = parser.parse_args()

    report = run_scenario(args.preset, args.scenario, args.events, args.seed, args.outdir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
