#!/usr/bin/env python3
"""Run the full evidence pipeline (benchmarks, classifiers, plots)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMMANDS = [
    ["uv", "run", "python", "benchmarks/run_dataset_suite.py", "--presets", "tuned_v2", "auto_tuned", "aggressive", "conservative", "energy_saver", "--out", "data/results/dataset_suite_extended.csv", "--logdir", "data/results/dataset_runs_extended"],
    ["uv", "run", "python", "benchmarks/run_ablation_study.py"],
    ["uv", "run", "python", "benchmarks/bootstrap_metrics.py", "data/results/dataset_runs_extended/breast_cancer_custom_breast_probe.json", "data/results/dataset_runs_extended/heart_disease_custom_health_hd82.json", "--out", "data/results/bootstrap_summary.json", "--samples", "500"],
    ["uv", "run", "python", "benchmarks/layer_classifier.py", "data/results/dataset_runs_extended/financial_aggressive.json", "data/results/dataset_runs_extended/network_security_aggressive.json", "data/results/dataset_runs_extended/iot_sensors_auto_tuned.json", "data/results/dataset_runs_extended/mitbih_ecg_auto_tuned.json", "--out", "data/results/layered_precision_full.csv"],
    ["uv", "run", "python", "benchmarks/plot_layered_precision.py", "--out", "assets/layered_precision.png"],
]


def main() -> None:
    for cmd in COMMANDS:
        print("\n>>>", " ".join(cmd))
        subprocess.check_call(cmd, cwd=ROOT)


if __name__ == "__main__":
    main()
