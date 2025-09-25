#!/usr/bin/env python3
"""Run targeted ablations to quantify feature/preset contributions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.run_dataset_suite import DATASETS, DatasetSpec, run_dataset
from sundew.config_presets import get_preset

AblationSpec = Tuple[str, str, Dict[str, float], str]

ABLATONS: Iterable[AblationSpec] = (
    (
        "breast_cancer",
        "custom_breast_probe",
        {"probe_every": 0},
        "no_probes",
    ),
    (
        "breast_cancer",
        "custom_breast_probe",
        {"w_anomaly": 0.35, "w_context": 0.20, "w_magnitude": 0.35, "w_urgency": 0.10},
        "reweighed_significance",
    ),
    (
        "heart_disease",
        "custom_health_hd82",
        {"energy_pressure": 0.03},
        "higher_pressure",
    ),
    (
        "iot_sensors",
        "auto_tuned",
        {"activation_threshold": 0.60},
        "higher_threshold",
    ),
)


def run_ablation(spec: AblationSpec, limit: int | None = None) -> Dict[str, object]:
    dataset_name, preset_name, overrides, label = spec
    dataset_spec: DatasetSpec = DATASETS[dataset_name]
    cfg = get_preset(preset_name)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    combined, raw = run_dataset(dataset_spec, preset_name, cfg, limit)
    combined["ablation_label"] = label
    return {"summary": combined, "raw": raw}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--out", type=Path, default=Path("data/results/ablation_study.csv"))
    parser.add_argument("--logdir", type=Path, default=Path("data/results/ablation_runs"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    args.logdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in ABLATONS:
        result = run_ablation(spec, args.limit)
        summary = result["summary"]
        raw = result["raw"]
        rows.append({k: v for k, v in summary.items() if k not in {"config", "report"}})
        log_path = args.logdir / f"{summary['dataset_name']}_{summary['preset']}_{summary['ablation_label']}.json"
        with log_path.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(
            f"[ablation] {summary['dataset_name']} | preset={summary['preset']} | "
            f"label={summary['ablation_label']} | recall={summary['recall']:.3f} | "
            f"savings={summary['estimated_energy_savings_pct']:.1f}%"
        )

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved ablation summary: {args.out}")


if __name__ == "__main__":
    main()
