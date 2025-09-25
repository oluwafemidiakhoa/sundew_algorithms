#!/usr/bin/env python3
"""Run PipelineRuntime on a dataset to capture probe instrumentation metrics."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import json

import pandas as pd

from sundew import build_simple_runtime
from sundew.config_presets import get_preset

FEATURE_KEYS = ("magnitude", "anomaly_score", "context_relevance", "urgency")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PipelineRuntime on a dataset CSV")
    parser.add_argument("dataset", type=Path, help="Path to CSV")
    parser.add_argument("--preset", default="custom_breast_probe", help="Preset to use")
    parser.add_argument("--log", type=Path, default=None, help="Optional JSON log output path")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    runtime = build_simple_runtime(get_preset(args.preset))

    records = []

    for idx, row in df.iterrows():
        event: Dict[str, float] = {k: float(row[k]) for k in FEATURE_KEYS if k in row}
        if "probe_hint" in row:
            event["probe_hint"] = float(row["probe_hint"])
        result = runtime.process(event)
        if args.log:
            records.append(
                {
                    "index": int(idx),
                    "activated": bool(result.activated),
                    "probe_hint": bool(event.get("probe_hint", 0.0)),
                }
            )

    report = runtime.report()
    print(f"preset={args.preset}")
    for key in (
        "samples_processed",
        "samples_activated",
        "activation_rate",
        "energy_savings_pct",
        "probe_activations",
    ):
        print(f"{key}: {report.get(key)}")

    if args.log:
        payload = {"records": records, "report": report}
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(json.dumps(payload, indent=2))
        print(f"Wrote runtime log to {args.log}")


if __name__ == "__main__":
    main()
