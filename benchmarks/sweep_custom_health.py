#!/usr/bin/env python3
"""Sweep custom_health overrides to map recall vs energy for health datasets."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sundew.config_presets import get_preset
from benchmarks.run_dataset_suite import DATASETS, DatasetSpec, run_dataset

DATASET_NAMES = ["breast_cancer", "heart_disease"]

PARAM_GRID: Dict[str, Iterable[Any]] = {
    "activation_threshold": [0.55, 0.60],
    "target_activation_rate": [0.15, 0.18],
    "energy_pressure": [0.02, 0.03, 0.04],
    "gate_temperature": [0.18, 0.15],
    "max_threshold": [0.85, 0.88],
}


def _iterate_configs() -> Iterable[Dict[str, Any]]:
    keys = list(PARAM_GRID.keys())
    for values in itertools.product(*(PARAM_GRID[k] for k in keys)):
        overrides = dict(zip(keys, values))
        signature_parts = [f"{k}={v}" for k, v in overrides.items()]
        overrides["__signature"] = "+".join(signature_parts)
        yield overrides


def main() -> None:
    out_path = Path("data/results/health_sweep_grid.csv")
    log_dir = Path("data/results/health_sweep_runs")
    log_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for dataset_name in DATASET_NAMES:
        spec: DatasetSpec = DATASETS[dataset_name]
        for overrides in _iterate_configs():
            cfg = get_preset("custom_health")
            for field, value in overrides.items():
                if field.startswith("__"):
                    continue
                setattr(cfg, field, value)
            combined, raw = run_dataset(spec, "custom_health", cfg)
            combined["override_signature"] = overrides["__signature"]
            rows.append({k: v for k, v in combined.items() if k not in {"config", "report"}})

            log_path = log_dir / (
                f"{dataset_name}_custom_health_"
                + overrides["__signature"].replace("=", "-").replace("+", "_")
                + ".json"
            )
            log_path.write_text(json.dumps({"summary": combined, "raw": raw}, indent=2))
            print(
                f"[{dataset_name}] {overrides['__signature']} | recall={combined['recall']:.3f} "
                f"| savings={combined['estimated_energy_savings_pct']:.1f}%"
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved sweep CSV: {out_path}")


if __name__ == "__main__":
    main()
