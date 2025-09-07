# benchmarks/plot_real_run.py
from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to JSON with y_true and y_pred arrays")
    ap.add_argument("--out", default="results/plots_real", help="Output directory for the plot")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(args.json, "r", encoding="utf-8") as f:
        out = json.load(f)

    y_true = out["y_true"]
    y_pred = out["y_pred"]
    n = min(len(y_true), 500)

    plt.figure(figsize=(12, 3))
    plt.plot(range(n), y_true[:n], lw=1, label="Ground truth (important)")
    plt.plot(range(n), y_pred[:n], lw=1, label="Sundew activation")
    plt.title("ECG: Ground Truth vs Sundew Activation (first 500)")
    plt.xlabel("Event index")
    plt.ylabel("Binary")
    plt.legend()
    p = os.path.join(args.out, "ecg_truth_vs_activation.png")
    plt.tight_layout()
    plt.savefig(p, dpi=150)
    print(f"saved {p}")


if __name__ == "__main__":
    main()
