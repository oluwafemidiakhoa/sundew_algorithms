#!/usr/bin/env python3
"""Template for capturing power measurements alongside Sundew events."""
from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path


def read_power_sample(simulate: bool = False) -> float:
    """Return instantaneous watts (simulated if requested)."""
    if simulate:
        return random.uniform(2.5, 4.5)
    raise NotImplementedError("Implement power capture for target hardware or use --simulate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Power capture template")
    parser.add_argument("--events", type=int, default=1000, help="Number of Sundew events processed")
    parser.add_argument("--out", type=Path, default=Path("data/results/hardware/power_trace.csv"))
    parser.add_argument("--interval", type=float, default=0.01, help="Sampling interval (seconds)")
    parser.add_argument("--simulate", action="store_true", help="Generate synthetic power samples")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "event_index", "watts"])
        for idx in range(args.events):
            watts = read_power_sample(simulate=args.simulate)
            writer.writerow([time.time(), idx, watts])
            time.sleep(args.interval)

    print(f"Wrote power trace to {args.out} (simulate={args.simulate})")


if __name__ == "__main__":
    main()
