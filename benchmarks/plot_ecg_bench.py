#!/usr/bin/env python3
# tools/plot_ecg_bench.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt  # pip install "sundew-algorithms[viz]"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot baseline vs actual energy from Sundew ECG benchmark JSON."
    )
    ap.add_argument("--json", required=True, help="Path to results JSON (from --save).")
    ap.add_argument("--out", default="", help="Optional output path (PNG/SVG).")
    ns = ap.parse_args()

    data = json.loads(Path(ns.json).read_text(encoding="utf-8"))
    rep = data.get("report", {})

    baseline = float(rep.get("baseline_energy_cost", 0.0))
    actual = float(rep.get("actual_energy_cost", 0.0))
    savings = float(rep.get("estimated_energy_savings_pct", 0.0))
    rate = float(rep.get("activation_rate", 0.0))
    activations = int(rep.get("activations", 0))

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = ["Baseline\n(cost)", "Actual\n(cost)"]
    ys = [baseline, actual]
    ax.bar(xs, ys)
    ax.set_ylabel("Energy cost (arbitrary units)")

    # Title now includes activations
    title = (
        f"Sundew ECG: {savings:.2f}% savings | rate {rate:.3f} | "
        f"activations {activations:,}"
    )
    ax.set_title(title)

    # Value labels on bars
    for i, v in enumerate(ys):
        ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom")

    fig.tight_layout()
    out = Path(ns.out) if ns.out else Path(ns.json).with_suffix(".png")
    fig.savefig(out, dpi=160)
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
