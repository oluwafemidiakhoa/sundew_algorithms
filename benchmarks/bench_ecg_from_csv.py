#!/usr/bin/env python3
"""
bench_ecg_from_csv.py

Run the Sundew gating algorithm on an ECG CSV (e.g., MIT-BIH Arrhythmia Database.csv)
and report activation rate & energy savings. Optionally save a JSON report.

Examples (PowerShell):

py -3.13 -m benchmarks.bench_ecg_from_csv `
  --csv "data\\MIT-BIH Arrhythmia Database.csv" `
  --limit 50000 `
  --activation-threshold 0.70 `
  --target-rate 0.12 `
  --gate-temperature 0.07 `
  --energy-pressure 0.04 `
  --max-threshold 0.92 `
  --refractory 0 `
  --save results\\ecg_bench_50000.json
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from sundew.config import SundewConfig
from sundew.core import ProcessingResult, SundewAlgorithm


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _to_plain(obj: object) -> Dict[str, Any]:
    """Dataclass-safe serializer; falls back to __dict__."""
    if is_dataclass(obj):
        return asdict(obj)  # type: ignore[arg-type]
    d = getattr(obj, "__dict__", {})
    return dict(d) if isinstance(d, dict) else {}


def _urgency_from_label(lbl: str) -> float:
    """
    Map MIT-BIH-like beat labels to a rough urgency score.
    This is heuristic and only used to produce richer demo events.
    """
    table = {
        "N": 0.10,  # normal beat
        "L": 0.40,  # left bundle branch block beat
        "R": 0.50,  # right bundle branch block beat
        "A": 0.60,  # atrial premature beat
        "V": 0.80,  # premature ventricular contraction
        "F": 0.80,  # fusion of ventricular and normal beat
        "E": 0.90,  # ventricular escape
        "!": 0.90,  # paced beat / artifact marker in some sets
        "/": 0.70,  # paced beat
        "j": 0.30,  # nodal junctional
        "?": 0.20,  # unknown
    }
    return table.get(lbl.strip(), 0.60)


def _anomaly_from_morph(row: Dict[str, str]) -> float:
    """
    Combine available QRS morphology features into [0, 1].
    Looks for columns containing 'qrs_morph' and averages |value|.
    """
    vals: List[float] = []
    for k, v in row.items():
        if "qrs_morph" in k:
            vals.append(abs(_safe_float(v)))
    if not vals:
        return 0.0
    avg = sum(vals) / float(len(vals))
    # Values in MIT-BIH CSV are already small; clamp to [0, 1]
    return max(0.0, min(1.0, avg))


def _row_to_event(row: Dict[str, str], id_col: str, label_col: str, rpeak_col: str) -> Dict[str, Any]:
    """
    Create a Sundew event dict from a CSV row.

    Keys used by Sundew's gating:
      - magnitude: scaled |rPeak|
      - anomaly_score: from qrs_morph columns
      - context_relevance: 0.8 if label != 'N' else 0.2
      - urgency: label-mapped urgency
      - type: label/category string (for display)
      - id: record/row identifier (optional)
    """
    label = row.get(label_col, "N")
    rpeak = abs(_safe_float(row.get(rpeak_col, 0.0)))
    magnitude = max(0.0, min(100.0, rpeak * 100.0))  # gating expects ~[0..100] before internal /100
    anomaly = _anomaly_from_morph(row)
    context = 0.8 if (label or "N") != "N" else 0.2
    urg = _urgency_from_label(str(label))

    return {
        "id": row.get(id_col, ""),
        "type": str(label or "N"),
        "magnitude": magnitude,
        "anomaly_score": anomaly,
        "context_relevance": context,
        "urgency": urg,
    }


def _iter_rows(
    csv_path: Path, delimiter: str, encoding: str, limit: int | None
) -> Iterable[Dict[str, str]]:
    with csv_path.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        count = 0
        for row in reader:
            yield row
            count += 1
            if limit is not None and count >= limit:
                break


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run Sundew on an ECG CSV and report energy/activation.")
    ap.add_argument("--csv", required=True, help="Path to the ECG CSV (e.g., MIT-BIH Arrhythmia Database.csv)")
    ap.add_argument("--limit", type=int, default=None, help="Max rows to process (default: all)")
    ap.add_argument("--delimiter", type=str, default=",", help="CSV delimiter (default: ',')")
    ap.add_argument("--encoding", type=str, default="utf-8", help="CSV file encoding (default: utf-8)")

    # Column hints (the defaults match the shared MIT-BIH CSV layout in this repo)
    ap.add_argument("--id-col", type=str, default="record", help="Column holding record ID (for display)")
    ap.add_argument("--label-col", type=str, default="type", help="Column holding beat label (e.g., N, V, A, ...)")
    ap.add_argument("--rpeak-col", type=str, default="0_rPeak", help="Column used as primary magnitude source")

    # Sundew tuning knobs
    ap.add_argument("--activation-threshold", type=float, default=0.70, dest="activation_threshold")
    ap.add_argument("--target-rate", type=float, default=0.12, dest="target_activation_rate")
    ap.add_argument("--gate-temperature", type=float, default=0.07, dest="gate_temperature")
    ap.add_argument("--energy-pressure", type=float, default=0.04, dest="energy_pressure")
    ap.add_argument("--max-threshold", type=float, default=0.92, dest="max_threshold")
    ap.add_argument("--refractory", type=int, default=0, dest="refractory")

    # Optional output
    ap.add_argument("--save", type=str, default="", help="Optional path to save JSON report")

    ns = ap.parse_args(argv)

    csv_path = Path(ns.csv)
    if not csv_path.exists():
        raise SystemExit(f"[bench] CSV not found: {csv_path}")

    # Configure Sundew
    cfg = SundewConfig(
        activation_threshold=ns.activation_threshold,
        target_activation_rate=ns.target_activation_rate,
        gate_temperature=ns.gate_temperature,
        energy_pressure=ns.energy_pressure,
        max_threshold=ns.max_threshold,
        refractory=ns.refractory,
    )
    cfg.validate()
    algo = SundewAlgorithm(cfg)

    # Read & run
    rows = list(_iter_rows(csv_path, ns.delimiter, ns.encoding, ns.limit))
    n = len(rows)
    print(f"[bench] Using delimiter='{ns.delimiter}', column='{ns.id_col}', n={n}")
    print("== Sundew ECG benchmark ==")

    processed: List[ProcessingResult] = []
    for row in rows:
        event = _row_to_event(row, ns.id_col, ns.label_col, ns.rpeak_col)
        res = algo.process(event)
        if res is not None:
            processed.append(res)

    report: Dict[str, Any] = algo.report()
    # Nicely aligned console report (mirrors your prior output)
    def _pf(k: str, v: Any) -> None:
        if isinstance(v, float):
            print(f"{k:26s}: {v:12.3f}")
        else:
            print(f"{k:26s}: {v}")

    # Keep key order similar to your logs
    _pf("total_inputs", n)
    _pf("activations", len(processed))
    _pf("activation_rate", report.get("activation_rate", 0.0))
    _pf("ema_activation_rate", report.get("ema_activation_rate", 0.0))
    _pf("total_energy_spent", report.get("total_energy_spent", 0.0))
    _pf("energy_remaining", report.get("energy_remaining", 0.0))
    _pf("threshold", report.get("threshold", 0.0))
    _pf("baseline_energy_cost", report.get("baseline_energy_cost", 0.0))
    _pf("actual_energy_cost", report.get("actual_energy_cost", 0.0))
    # The savings in your logs was printed with fewer spaces; keep that label verbatim:
    key = "estimated_energy_savings_pct"
    val = float(report.get(key, 0.0))
    print(f"{key}: {val:12.3f}")

    # Optional JSON save
    if ns.save:
        out = {
            "config": _to_plain(cfg),
            "report": report,
            # Keep it light: we don’t dump every processed event by default here
        }
        p = Path(ns.save)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[bench] saved JSON to {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
