#!/usr/bin/env python3
"""Narrow sweep around custom_health for heart_disease."""
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
    "activation_threshold": [0.55, 0.56, 0.57, 0.58],
    "target_activation_rate": [0.14, 0.15, 0.16],
    "energy_pressure": [0.02, 0.025, 0.03],
    "gate_temperature": [0.14, 0.15, 0.16],
    "max_threshold": [0.87, 0.88],
}


def main() -> None:
    spec = DATASETS["heart_disease"]
    rows: list[Dict[str, Any]] = []
    log_dir = Path("data/results/health_sweep_narrow_runs")
    log_dir.mkdir(parents=True, exist_ok=True)

    for combo in itertools.product(*PARAM_GRID.values()):
        overrides = dict(zip(PARAM_GRID.keys(), combo))
        cfg = get_preset("custom_health")
        for field, value in overrides.items():
            setattr(cfg, field, value)
        combined, raw = run_dataset(spec, "custom_health", cfg)
        signature = "+".join(f"{k}={value}" for k, value in overrides.items())
        combined["override_signature"] = signature
        rows.append({k: v for k, v in combined.items() if k not in {"config", "report"}})

        log_path = log_dir / (
            "heart_disease_custom_health_"
            + signature.replace("=", "-").replace("+", "_")
            + ".json"
        )
        log_path.write_text(json.dumps({"summary": combined, "raw": raw}, indent=2))
        print(
            f"{signature} | recall={combined['recall']:.3f} | "
            f"savings={combined['estimated_energy_savings_pct']:.2f}% | "
            f"activation={combined['activation_rate']:.3f}"
        )

    out_path = Path("data/results/health_sweep_narrow.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved narrow sweep CSV: {out_path}")


if __name__ == "__main__":
    main()
