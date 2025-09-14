#!/usr/bin/env python3
# tools/plot_ecg_bench.py
from __future__ import annotations
import json
from pathlib import Path
import argparse
import matplotlib.pyplot as plt  # pip install "sundew-algorithms[viz]"

def main() -> None:
    ap = argparse.ArgumentParser(description="Plot baseline vs actual energy from Sundew ECG benchmark JSON.")
    ap.add_argument("--json", required=True, help="Path to results JSON (from --save).")
    ap.add_argument("--out", default="", help="Optional PNG path (default: alongside JSON).")
    ns = ap.parse_args()

    data = json.loads(Path(ns.json).read_text(encoding="utf-8"))
    rep = data["report"]
    baseline = float(rep["baseline_energy_cost"])
    actual = float(rep["actual_energy_cost"])
    savings = float(rep["estimated_energy_savings_pct"])
    rate = float(rep["activation_rate"])

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = ["Baseline\n(cost)", "Actual\n(cost)"]
    ys = [baseline, actual]
    ax.bar(xs, ys)
    ax.set_ylabel("Energy cost (arbitrary units)")
    ax.set_title(f"Sundew ECG: savings≈{savings:.2f}%  |  activation≈{rate:.3f}")
    for i, v in enumerate(ys):
        ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom")

    fig.tight_layout()
    out = Path(ns.out) if ns.out else Path(ns.json).with_suffix(".png")
    fig.savefig(out, dpi=160)
    print(f"[plot] wrote {out}")

if __name__ == "__main__":
    main()
