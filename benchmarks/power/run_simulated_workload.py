
"""Simulated power benchmark harness for Sundew Core SDK.

This placeholder workload generates synthetic feature streams and records
activation decisions along with estimated power consumption. It provides a
quick feedback loop while hardware instrumentation is being prepared.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

from sundew_core_sdk.config import SDKConfig
from sundew_core_sdk.controller import AdaptiveGateController
from sundew_core_sdk.metrics import MetricsTracker


def generate_features(seed: int, count: int, profile: str = "balanced") -> List[Dict[str, float]]:
    random.seed(seed)
    features: List[Dict[str, float]] = []
    profile = profile.lower()
    activation_bias = {"aggressive": 0.15, "balanced": 0.05, "conservative": 0.02}.get(profile, 0.05)
    slope_scale = {"aggressive": 4.5, "balanced": 3.0, "conservative": 1.8}.get(profile, 3.0)
    variability_scale = {"aggressive": 1.0, "balanced": 0.8, "conservative": 0.5}.get(profile, 0.8)
    for idx in range(count):
        base = 120 + 40 * random.random()
        slope = random.uniform(-slope_scale, slope_scale)
        activity = random.random()
        meal = 1.0 if random.random() < activation_bias else 0.0
        features.append(
            {
                "timestamp": idx,
                "glucose_mgdl": base,
                "roc_mgdl_min": slope,
                "iob_proxy": meal * random.uniform(0.0, 2.0),
                "cob_proxy": meal * random.uniform(0.0, 80.0),
                "activity_factor": activity,
                "variability": random.uniform(0.0, variability_scale),
                "deviation": random.uniform(-20.0, 20.0),
                "urgency": random.random(),
                "magnitude": base,
                "anomaly_score": random.random(),
                "context": activity,
            }
        )
    return features


def estimate_power(activated: bool) -> float:
    base_idle = 1.2  # watts
    inference_cost = 4.8  # incremental watts when heavy workload fires
    return base_idle + (inference_cost if activated else 0.0)


def run_simulation(samples: int, seed: int, profile: str) -> pd.DataFrame:
    cfg = SDKConfig()
    if profile == "aggressive":
        cfg.target_activation = 0.60
    elif profile == "balanced":
        cfg.target_activation = 0.25
    else:
        cfg.target_activation = 0.15
    controller = AdaptiveGateController(cfg)
    controller.load_native()

    rows: List[Dict[str, float]] = []
    tracker = MetricsTracker()
    features = generate_features(seed, samples, profile=profile)
    exported: List[Dict[str, float]] = []
    start = time.perf_counter()
    for event in features:
        activated = controller.decide(event)
        power = estimate_power(activated)
        tracker.record(activated, power)
        exported.append(
            {
                "timestamp": event["timestamp"],
                "activated": int(activated),
                "power_w": power,
                "profile": profile,
            }
        )
        rows.append(
            {
                "timestamp": event["timestamp"],
                "activated": float(activated),
                "power_w": power,
                "glucose_mgdl": event["glucose_mgdl"],
            }
        )
    runtime = time.perf_counter() - start
    df = pd.DataFrame(rows)
    df.attrs["runtime_seconds"] = runtime
    df.attrs["metrics"] = tracker.snapshot()
    df.attrs["exported"] = exported
    return df


def summarize(df: pd.DataFrame) -> Dict[str, float]:
    activation_rate = float(df["activated"].mean())
    avg_power = float(df["power_w"].mean())
    idle_power_series = df.loc[df["activated"] == 0, "power_w"]
    idle_power = float(idle_power_series.mean()) if not idle_power_series.empty else 1.2
    heavy_power = idle_power + 4.8
    estimated_savings = 1.0 - avg_power / heavy_power
    return {
        "activation_rate": activation_rate,
        "avg_power": avg_power,
        "baseline_idle": idle_power,
        "baseline_heavy": heavy_power,
        "estimated_savings": estimated_savings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulated power benchmark")
    parser.add_argument("--samples", type=int, default=5000, help="number of events to simulate")
    parser.add_argument("--seed", type=int, default=17, help="random seed")
    parser.add_argument("--export", type=Path, help="optional path to export metrics JSON")
    parser.add_argument("--profile", choices=["aggressive", "balanced", "conservative"], default="balanced", help="activation profile")
    args = parser.parse_args()

    df = run_simulation(args.samples, args.seed, args.profile)
    metrics = df.attrs.get("metrics")
    summary = summarize(df)
    if args.export:
        metrics = df.attrs.get("metrics")
        exported = df.attrs.get("exported", [])
        payload = {
            "profile": args.profile,
            "summary": summary,
            "metrics": metrics.as_dict() if metrics else {},
            "events": exported,
        }
        args.export.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")
    print("Runtime (s):", round(df.attrs.get("runtime_seconds", 0.0), 4))
    if metrics is not None:
        print("Tracker activation rate:", f"{metrics.activation_rate:.4f}")
        print("Tracker avg power:", f"{metrics.avg_power_w:.4f}")
    print("profile:", args.profile)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
