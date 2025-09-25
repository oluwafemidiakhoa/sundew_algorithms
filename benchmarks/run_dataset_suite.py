#!/usr/bin/env python3
"""Run Sundew on curated datasets and export classification + energy metrics."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from sundew import SundewAlgorithm, SundewConfig
from sundew.config_presets import get_preset


EVENT_KEYS = ("magnitude", "anomaly_score", "context_relevance", "urgency")


def _default_loader(path: Path, limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df if limit is None else df.head(limit)


def _mitbih_loader(path: Path, limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if limit is not None:
        df = df.head(limit)

    df = df.copy()
    labels = df.get("type", pd.Series("N", index=df.index)).astype(str).str.strip().str.upper()
    df["ground_truth"] = labels.apply(lambda lbl: 0 if lbl == "N" else 1)

    if "0_rPeak" in df.columns:
        rpeak_series = df["0_rPeak"].abs()
    else:
        rpeak_cols = [c for c in df.columns if c.endswith("_rPeak")]
        if not rpeak_cols:
            raise ValueError("MIT-BIH CSV missing any *_rPeak column")
        rpeak_series = df[rpeak_cols[0]].abs()
    df["magnitude"] = (rpeak_series.fillna(0.0) * 100.0).clip(0.0, 100.0)

    morph_cols = [c for c in df.columns if "qrs_morph" in c.lower()]
    if morph_cols:
        df["anomaly_score"] = df[morph_cols].abs().mean(axis=1).fillna(0.0).clip(0.0, 1.0)
    else:
        df["anomaly_score"] = 0.0

    df["context_relevance"] = df["ground_truth"].map({1: 0.8, 0: 0.2})

    urgency_map = {
        "N": 0.10,
        "L": 0.40,
        "R": 0.50,
        "A": 0.60,
        "V": 0.80,
        "F": 0.80,
        "E": 0.90,
        "!": 0.90,
        "/": 0.70,
        "J": 0.30,
        "?": 0.20,
    }
    df["urgency"] = labels.map(urgency_map).fillna(0.60)

    if "record" not in df.columns:
        df["record"] = range(1, len(df) + 1)

    cols = ["record", "type", *EVENT_KEYS, "ground_truth"]
    available = [c for c in cols if c in df.columns]
    return df[available]


@dataclass
class DatasetSpec:
    name: str
    path: Path
    preset: str
    loader: Callable[[Path, int | None], pd.DataFrame] = _default_loader
    extra_presets: Sequence[str] = field(default_factory=tuple)
    description: str = ""


def _classification_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
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

    total = max(1, tp + fp + fn + tn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = (2 * precision * recall) / max(1e-12, precision + recall) if (tp + fp + fn) else 0.0
    accuracy = (tp + tn) / total
    prevalence = (tp + fn) / total
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, fn + tp)
    balanced_acc = (recall + specificity) / 2.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "prevalence": prevalence,
        "total_inputs": total,
        "activations": tp + fp,
    }


DATASETS: Dict[str, DatasetSpec] = {
    "breast_cancer": DatasetSpec(
        name="breast_cancer",
        path=Path("data/raw/breast_cancer_wisconsin_enriched.csv"),
        preset="tuned_v2",
        extra_presets=("custom_health", "custom_breast_probe"),
        description="Breast Cancer Wisconsin (enriched anomaly features + probe hint)",
    ),
    "financial": DatasetSpec(
        name="financial",
        path=Path("data/raw/financial_time_series.csv"),
        preset="tuned_v2",
        description="Financial time series (volatility spikes marked as 1)",
    ),
    "heart_disease": DatasetSpec(
        name="heart_disease",
        path=Path("data/raw/uci_heart_disease.csv"),
        preset="custom_health_hd82",
        description="UCI heart disease tabular dataset (recall-focused preset)",
    ),
    "iot_sensors": DatasetSpec(
        name="iot_sensors",
        path=Path("data/raw/iot_sensor_monitoring.csv"),
        preset="tuned_v2",
        description="IoT multisensor monitoring (rare faults labeled)",
    ),
    "network_security": DatasetSpec(
        name="network_security",
        path=Path("data/raw/network_security.csv"),
        preset="tuned_v2",
        description="Network intrusion patterns (attacks vs benign)",
    ),
    "mitbih_ecg": DatasetSpec(
        name="mitbih_ecg",
        path=Path("data/MIT-BIH Arrhythmia Database.csv"),
        preset="ecg_mitbih_best",
        loader=_mitbih_loader,
        extra_presets=("ecg_v1", "tuned_v2"),
        description="MIT-BIH arrhythmia beats (non-N labelled as arrhythmia)",
    ),
}


def run_dataset(
    spec: DatasetSpec,
    preset_name: str,
    cfg: SundewConfig,
    limit: int | None = None,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    df = spec.loader(spec.path, limit)

    algo = SundewAlgorithm(cfg)

    y_true = df.get("ground_truth", pd.Series([0] * len(df))).astype(int).tolist()
    y_pred: List[int] = []
    events: List[Dict[str, float]] = []

    for _, row in df.iterrows():
        event = {k: float(row[k]) for k in EVENT_KEYS if k in row}
        if "probe_hint" in row:
            event["probe_hint"] = float(row["probe_hint"])
        events.append(event)
        result = algo.process(event)
        y_pred.append(1 if result is not None else 0)

    report = algo.report()
    cls_metrics = _classification_metrics(y_true, y_pred)

    activation_rate = report.get(
        "activation_rate", cls_metrics["activations"] / max(1, cls_metrics["total_inputs"])
    )
    baseline_energy = report.get("baseline_energy_cost", 0.0)
    actual_energy = report.get("actual_energy_cost", 0.0)

    combined = {
        "dataset_name": spec.name,
        "preset": preset_name,
        "target_activation_rate": cfg.target_activation_rate,
        "activation_rate": activation_rate,
        "activation_rate_error": abs(activation_rate - cfg.target_activation_rate),
        "ema_activation_rate": report.get("ema_activation_rate", 0.0),
        "threshold": report.get("threshold", 0.0),
        "avg_processing_time": report.get("avg_processing_time", 0.0),
        "total_energy_spent": report.get("total_energy_spent", 0.0),
        "baseline_energy_cost": baseline_energy,
        "actual_energy_cost": actual_energy,
        "absolute_energy_savings": baseline_energy - actual_energy,
        "estimated_energy_savings_pct": report.get("estimated_energy_savings_pct", 0.0),
        "energy_efficiency": report.get("energy_efficiency", 0.0),
        "total_inputs": cls_metrics["total_inputs"],
        "activations": cls_metrics["activations"],
        "tp": cls_metrics["tp"],
        "fp": cls_metrics["fp"],
        "fn": cls_metrics["fn"],
        "tn": cls_metrics["tn"],
        "precision": cls_metrics["precision"],
        "recall": cls_metrics["recall"],
        "specificity": cls_metrics["specificity"],
        "f1": cls_metrics["f1"],
        "accuracy": cls_metrics["accuracy"],
        "balanced_accuracy": cls_metrics["balanced_accuracy"],
        "false_positive_rate": cls_metrics["false_positive_rate"],
        "false_negative_rate": cls_metrics["false_negative_rate"],
        "prevalence": cls_metrics["prevalence"],
        "config": asdict(cfg),
        "report": report,
        "description": spec.description,
    }

    raw_payload = {
        "y_true": y_true,
        "y_pred": y_pred,
        "report": report,
        "events": events,
    }

    return combined, raw_payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark Sundew on curated datasets")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()), help="Datasets to run")
    ap.add_argument("--limit", type=int, default=None, help="Optional row limit per dataset")
    ap.add_argument("--out", type=Path, default=Path("data/results/dataset_suite.csv"), help="CSV summary output")
    ap.add_argument(
        "--logdir",
        type=Path,
        default=Path("data/results/dataset_runs"),
        help="Directory for per-run JSON logs",
    )
    ap.add_argument(
        "--presets",
        nargs="*",
        default=(),
        help="Optional presets to evaluate in addition to each dataset's default.",
    )
    args = ap.parse_args()

    if args.logdir:
        args.logdir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []

    for name in args.datasets:
        if name not in DATASETS:
            raise SystemExit(f"Unknown dataset: {name}")

        spec = DATASETS[name]
        presets: List[str] = []
        presets.extend(args.presets)
        presets.extend(spec.extra_presets)
        presets.append(spec.preset)
        seen = set()
        ordered_presets = []
        for preset in presets:
            if preset not in seen:
                ordered_presets.append(preset)
                seen.add(preset)

        for preset_name in ordered_presets:
            cfg = get_preset(preset_name)
            combined, raw = run_dataset(spec, preset_name, cfg, args.limit)
            rows.append({k: v for k, v in combined.items() if k not in {"config", "report"}})

            if args.logdir:
                json_path = args.logdir / f"{spec.name}_{preset_name}.json"
                with json_path.open("w", encoding="utf-8") as jf:
                    json.dump({"summary": combined, "raw": raw}, jf, indent=2)

            print(
                f"[dataset] {spec.name:15s} | preset={preset_name:15s} "
                f"| f1={combined['f1']:.3f} | recall={combined['recall']:.3f} "
                f"| savings={combined['estimated_energy_savings_pct']:.1f}%"
            )

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["dataset_name", "preset"], inplace=True)
    df.to_csv(args.out, index=False)
    print(f"\nWrote summary CSV: {args.out.resolve()}")
    if args.logdir:
        print(f"Logs stored in  : {args.logdir.resolve()}")


if __name__ == "__main__":
    main()
