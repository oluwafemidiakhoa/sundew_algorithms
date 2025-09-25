#!/usr/bin/env python3
"""Bootstrap precision/recall/F1 metrics from dataset logs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def _bootstrap_metrics(y_true: np.ndarray, y_pred: np.ndarray, n: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    activations: List[float] = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        yt = y_true[idx]
        yp = y_pred[idx]
        tp = np.sum((yt == 1) & (yp == 1))
        fp = np.sum((yt == 0) & (yp == 1))
        fn = np.sum((yt == 1) & (yp == 0))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = (2 * precision * recall) / max(1e-12, precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        activations.append(np.mean(yp))
    def ci(x: List[float]) -> Dict[str, float]:
        arr = np.sort(x)
        lower = arr[int(0.025 * len(arr))]
        upper = arr[int(0.975 * len(arr))]
        return {"mean": float(np.mean(arr)), "low": float(lower), "high": float(upper)}
    return {
        "precision": ci(precisions),
        "recall": ci(recalls),
        "f1": ci(f1s),
        "activation_rate": ci(activations),
    }


def load_run(path: Path) -> Dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    raw = payload.get("raw", payload)
    y_true = np.array(raw["y_true"], dtype=int)
    y_pred = np.array(raw["y_pred"], dtype=int)
    energy = payload.get("summary", {}).get("estimated_energy_savings_pct")
    return {"y_true": y_true, "y_pred": y_pred, "energy": energy}


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap metrics from dataset run JSON")
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--samples", type=int, default=1000, help="Bootstrap iterations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/results/bootstrap_summary.json"),
    )
    args = parser.parse_args()

    out_data = {}
    for log in args.logs:
        run = load_run(log)
        metrics = _bootstrap_metrics(run["y_true"], run["y_pred"], args.samples, args.seed)
        metrics["mean_energy_savings"] = run["energy"]
        out_data[log.name] = metrics
        print(
            f"[bootstrap] {log.name}: precision {metrics['precision']['mean']:.3f} "
            f"(95% CI {metrics['precision']['low']:.3f}-{metrics['precision']['high']:.3f})"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(out_data, fh, indent=2)
    print(f"Saved bootstrap summary: {args.out}")


if __name__ == "__main__":
    main()
