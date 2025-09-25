#!/usr/bin/env python3
"""Tight sweep around custom_health_hd82 for heart disease."""
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
    "activation_threshold": [0.55, 0.56, 0.57],
    "target_activation_rate": [0.14, 0.15, 0.16],
    "energy_pressure": [0.019, 0.02, 0.021],
    "gate_temperature": [0.13, 0.14, 0.15],
    "max_threshold": [0.87, 0.88],
}


def main() -> None:
    spec = DATASETS["heart_disease"]
    log_dir = Path("data/results/heart_hd82_tight_runs")
    log_dir.mkdir(parents=True, exist_ok=True)
    rows: list[Dict[str, Any]] = []

    for values in itertools.product(*PARAM_GRID.values()):
        overrides = dict(zip(PARAM_GRID.keys(), values))
        cfg = get_preset("custom_health_hd82")
        for field, value in overrides.items():
            setattr(cfg, field, value)
        combined, raw = run_dataset(spec, "custom_health_hd82", cfg)
        signature = "+".join(f"{k}={value}" for k, value in overrides.items())
        combined["override_signature"] = signature
        rows.append({k: v for k, v in combined.items() if k not in {"config", "report"}})

        log_path = log_dir / (
            "heart_hd82_" + signature.replace("=", "-").replace("+", "_") + ".json"
        )
        log_path.write_text(json.dumps({"summary": combined, "raw": raw}, indent=2))
        print(
            f"{signature} | recall={combined['recall']:.3f} | "
            f"savings={combined['estimated_energy_savings_pct']:.2f}% | activation={combined['activation_rate']:.3f}"
        )

    out_path = Path("data/results/heart_hd82_tight.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved tight sweep CSV: {out_path}")


if __name__ == "__main__":
    main()
