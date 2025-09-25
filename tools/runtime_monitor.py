#!/usr/bin/env python3
"""Example listener wiring for Sundew runtime monitoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sundew import build_simple_runtime
from sundew.config_presets import get_preset


def main() -> None:
    parser = argparse.ArgumentParser(description="Runtime monitoring example")
    parser.add_argument("--preset", default="custom_breast_probe")
    parser.add_argument("--dataset", type=Path, help="CSV with event features")
    parser.add_argument("--log", type=Path, default=Path("data/results/runtime_monitor.json"))
    args = parser.parse_args()

    runtime = build_simple_runtime(get_preset(args.preset))
    records = []

    def listener(result, components):
        records.append(
            {
                "activated": result.activated,
                "energy_cost": components["energy"]["cost"],
                "threshold": components["control"]["threshold"],
            }
        )

    runtime.add_listener(listener)

    import pandas as pd  # type: ignore

    df = pd.read_csv(args.dataset)
    for _, row in df.iterrows():
        event = {k: float(row[k]) for k in ("magnitude", "anomaly_score", "context_relevance", "urgency") if k in row}
        if "probe_hint" in row:
            event["probe_hint"] = float(row["probe_hint"])
        runtime.process(event)

    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps({"records": records, "report": runtime.report()}, indent=2))
    print(f"Wrote runtime monitor log to {args.log}")


if __name__ == "__main__":
    main()
