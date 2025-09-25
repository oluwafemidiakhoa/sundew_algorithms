#!/usr/bin/env python3
"""Merge runtime probe log with power trace to estimate measured savings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge runtime log and power trace")
    parser.add_argument("--runtime", type=Path, default=Path("data/results/runtime_probe_log.json"))
    parser.add_argument("--power", type=Path, default=Path("data/results/hardware/power_trace.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/results/hardware/merged_probe_power.json"))
    args = parser.parse_args()

    runtime = json.loads(args.runtime.read_text())
    power_df = pd.read_csv(args.power)

    avg_watts = power_df["watts"].mean()
    total_energy = power_df["watts"].sum() * (power_df["timestamp"].diff().mean())

    summary = runtime.get("report", {})
    merged = {
        "runtime_report": summary,
        "probe_events": sum(1 for r in runtime.get("records", []) if r.get("probe_hint") and r.get("activated")),
        "average_watts": avg_watts,
        "approx_energy_joules": total_energy,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, indent=2))
    print(f"Merged report saved to {args.out}")


if __name__ == "__main__":
    main()
