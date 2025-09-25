#!/usr/bin/env python3
"""Plot baseline vs layered precision uplift for layered classifier runs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_FILES = [
    Path("data/results/layered_precision.csv"),
    Path("data/results/layered_precision_extended.csv"),
    Path("data/results/layered_precision_iot_mitbih.csv"),
]
DATASET_SUITE = Path("data/results/dataset_suite_extended.csv")


def _load_layered(files: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for f in files:
        if f.exists():
            frames.append(pd.read_csv(f))
    if not frames:
        raise SystemExit("No layered precision files found.")
    df = pd.concat(frames, ignore_index=True)
    df.rename(
        columns={
            "baseline_precision": "baseline",
            "layered_precision": "layered",
            "energy_savings_pct": "savings",
        },
        inplace=True,
    )
    return df


def _load_energy() -> pd.DataFrame:
    if not DATASET_SUITE.exists():
        return pd.DataFrame()
    df = pd.read_csv(DATASET_SUITE)
    return df[["dataset_name", "preset", "estimated_energy_savings_pct"]]


def plot_precision(df: pd.DataFrame, out: Path, energy: pd.DataFrame | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    datasets = df["dataset"].unique()
    datasets.sort()
    x = range(len(datasets))
    baseline = [df[df["dataset"] == d]["baseline"].mean() for d in datasets]
    layered = [df[df["dataset"] == d]["layered"].mean() for d in datasets]

    width = 0.35
    ax.bar([i - width / 2 for i in x], baseline, width, label="Baseline")
    ax.bar([i + width / 2 for i in x], layered, width, label="Layered")

    ax.set_ylabel("Precision")
    ax.set_title("Layered Classifier Precision Uplift")
    ax.set_xticks(list(x))
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()

    if energy is not None and not energy.empty:
        annotations = []
        for d in datasets:
            subset = df[df["dataset"] == d]
            preset = subset.iloc[0]["preset"]
            energy_row = energy[(energy["dataset_name"] == d) & (energy["preset"] == preset)]
            if not energy_row.empty:
                savings = energy_row.iloc[0]["estimated_energy_savings_pct"]
                annotations.append((d, savings))
        for idx, (dataset, savings) in enumerate(annotations):
            ax.text(idx, layered[idx] + 0.03, f"{savings:.1f}% energy", ha="center", fontsize=9)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved plot to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot layered precision uplift")
    parser.add_argument("--out", type=Path, default=Path("assets/layered_precision.png"))
    parser.add_argument("--csv", type=Path, nargs="*", default=DEFAULT_FILES)
    args = parser.parse_args()

    df = _load_layered(args.csv)
    energy = _load_energy()
    plot_precision(df, args.out, energy)


if __name__ == "__main__":
    main()
