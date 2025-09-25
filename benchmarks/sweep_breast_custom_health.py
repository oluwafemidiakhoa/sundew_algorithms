#!/usr/bin/env python3
"""Targeted tuning for breast cancer dataset."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sundew.config_presets import get_preset
from benchmarks.run_dataset_suite import DATASETS, run_dataset

PARAM_GRID: Dict[str, list[Any]] = {
    "activation_threshold": [0.52, 0.54, 0.56],
    "target_activation_rate": [0.16, 0.18],
    "energy_pressure": [0.02, 0.03],
    "gate_temperature": [0.18, 0.20],
    "max_threshold": [0.86, 0.88],
    "w_magnitude": [0.15],
    "w_anomaly": [0.50, 0.52],
    "w_context": [0.25, 0.28],
}


def main() -> None:
    spec = DATASETS["breast_cancer"]
    log_dir = Path("data/results/breast_cancer_tuning_runs")
    log_dir.mkdir(parents=True, exist_ok=True)
    rows: list[Dict[str, Any]] = []

    for combo in itertools.product(*PARAM_GRID.values()):
        overrides = dict(zip(PARAM_GRID.keys(), combo))
        w_urgency = 1.0 - (
            overrides["w_magnitude"] + overrides["w_anomaly"] + overrides["w_context"]
        )
        if w_urgency <= 0:
            continue

        cfg = get_preset("custom_health")
        for field, value in overrides.items():
            setattr(cfg, field, value)
        cfg.w_urgency = w_urgency

        combined, raw = run_dataset(spec, "custom_health", cfg)
        signature = "+".join(f"{k}={value}" for k, value in overrides.items()) + f"+w_urgency={w_urgency:.3f}"
        combined["override_signature"] = signature
        rows.append({k: v for k, v in combined.items() if k not in {"config", "report"}})

        log_path = log_dir / (
            "breast_custom_health_" + signature.replace("=", "-").replace("+", "_") + ".json"
        )
        log_path.write_text(json.dumps({"summary": combined, "raw": raw}, indent=2))
        print(
            f"{signature} | recall={combined['recall']:.3f} | "
            f"savings={combined['estimated_energy_savings_pct']:.2f}% | "
            f"activation={combined['activation_rate']:.3f}"
        )

    out_path = Path("data/results/breast_cancer_tuning.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved breast cancer tuning CSV: {out_path}")


if __name__ == "__main__":
    main()
