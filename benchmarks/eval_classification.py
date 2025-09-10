#!/usr/bin/env python3
# benchmarks/eval_classification.py
from __future__ import annotations

import argparse
import json
from typing import Dict, List


def _metrics_from_series(y_true: List[int], y_pred: List[int]) -> Dict[str, float | int]:
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
        else:
            tn += 1
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = (2 * prec * rec) / max(1e-12, prec + rec) if (tp + fp + fn) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "total_inputs": len(y_true),
        "activations": sum(1 for p in y_pred if p == 1),
    }


def _metrics_from_counts(counts: Dict[str, float | int]) -> Dict[str, float | int]:
    # Ensure derived fields exist/are consistent
    tp = int(counts.get("tp", 0))
    fp = int(counts.get("fp", 0))
    fn = int(counts.get("fn", 0))
    tn = int(counts.get("tn", 0))
    prec = float(counts.get("precision", tp / max(1, tp + fp)))
    rec = float(counts.get("recall", tp / max(1, tp + fn)))
    if "f1" in counts:
        f1 = float(counts["f1"])
    else:
        denom = max(1e-12, prec + rec)
        f1 = (2 * prec * rec) / denom if (tp + fp + fn) > 0 else 0.0
    total = int(counts.get("total_inputs", tp + fp + fn + tn))
    activ = int(counts.get("activations", tp + fp))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "total_inputs": total,
        "activations": activ,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Sundew classification output JSON.")
    ap.add_argument("--json", required=True, help="Path to run_ecg output JSON")
    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "y_true" in data and "y_pred" in data:
        y_true = [int(x) for x in data["y_true"]]
        y_pred = [int(x) for x in data["y_pred"]]
        counts = _metrics_from_series(y_true, y_pred)
    elif "counts" in data:
        counts = _metrics_from_counts(data["counts"])
    else:
        raise SystemExit("ERROR: JSON has neither 'y_true'/'y_pred' nor 'counts'.")

    # Pretty print
    print(f"total_inputs              : {counts['total_inputs']}")
    print(f"activations               : {counts['activations']}")
    print(f"precision                 : {counts['precision']:.6f}")
    print(f"recall                    : {counts['recall']:.6f}")
    print(f"f1                        : {counts['f1']:.6f}")
    print(
        f"tp/fp/fn/tn               : "
        f"{counts['tp']} / {counts['fp']} / {counts['fn']} / {counts['tn']}"
    )

    # If present, also echo a few energy/report fields for convenience
    rep = data.get("report", {})
    for k in (
        "activation_rate",
        "ema_activation_rate",
        "avg_processing_time",
        "total_energy_spent",
        "energy_remaining",
        "threshold",
        "estimated_energy_savings_pct",
    ):
        if k in rep:
            v = rep[k]
            if isinstance(v, float):
                print(f"{k:25s} : {v}")
            else:
                print(f"{k:25s} : {v}")


if __name__ == "__main__":
    main()
