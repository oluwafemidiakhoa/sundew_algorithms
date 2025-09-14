#!/usr/bin/env python3
# tools/plot_significance_hist.py
"""
Plot a histogram of 'significance' values from a Sundew demo JSON.

Usage:
  python tools/plot_significance_hist.py --json "%USERPROFILE%\\Downloads\\demo_run.json" --bins 24
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=False, default=r"%USERPROFILE%\Downloads\demo_run.json")
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    jpath = Path(os.path.expandvars(args.json))
    data = json.load(jpath.open("r", encoding="utf-8"))

    events: List[dict] = data.get("processed_events", [])
    if not events:
        raise SystemExit("No processed_events in JSON (nothing to plot).")

    sigs = [float(e["significance"]) for e in events]
    out_path = Path(args.out) if args.out else jpath.with_suffix("").with_name(jpath.stem + "_sig_hist.png")

    plt.figure()
    plt.hist(sigs, bins=args.bins)
    plt.title("Sundew: Significance histogram")
    plt.xlabel("significance")
    plt.ylabel("count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    print(f"Saved histogram → {out_path}")


if __name__ == "__main__":
    main()
