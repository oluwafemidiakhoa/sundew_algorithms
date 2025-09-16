#!/usr/bin/env python3
"""
Build a ranked scorecard from one or more *.summary.json files.

- Inputs may be individual files and/or directories.
- Directories are scanned recursively for files matching --pattern (default: *.summary.json).
- Designed to work on Windows CMD (no globbing), PowerShell, and shells that expand globs.

Scoring (all 0..1, higher is better):
  energy_eff  = clamp01((energy_recovered / max(1, energy_spent)) / 2)   # capped at ratio 2.0
  stability   = 1 - clamp01(oscillation_score)                           # lower oscillation is better
  cap_headroom= 1 - clamp01(time_at_cap_pct / 100)                       # less time at cap is better
  responsive  = 1 - clamp01(abs(activation_rate_reported_pct - target) / target)

Overall score = weighted sum of the above (weights adjustable via --weights).

Output JSON:
{
  "generated_at": "...",
  "weights": {...},
  "target_activation_pct": 20.0,
  "items": [
     {
       "name": "run_hot.txt.summary.json",
       "path": "runs/run_hot.txt.summary.json",
       "scores": { "overall": 0.8123, "energy_eff":..., "stability":..., "cap_headroom":..., "responsive":... },
       "raw": { ... selected raw fields ... },
       "rank": 1
     }, ...
  ]
}
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
from typing import Dict, List, Tuple

DEFAULT_PATTERN = "*.summary.json"

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    try:
        return a / b if b else default
    except Exception:
        return default

def find_inputs(paths: List[str], pattern: str) -> List[str]:
    files: List[str] = []
    if not paths:
        return files
    for p in paths:
        if any(ch in p for ch in ["*", "?", "["]):  # manual glob if user passed wildcards
            files.extend(glob.glob(p, recursive=True))
        elif os.path.isdir(p):
            # recursive scan with pattern
            for root, _dirs, _files in os.walk(p):
                files.extend(glob.glob(os.path.join(root, pattern)))
        else:
            files.append(p)
    # de-dupe, keep only existing files
    uniq = []
    seen = set()
    for f in files:
        f = os.path.normpath(f)
        if os.path.isfile(f) and f not in seen:
            seen.add(f)
            uniq.append(f)
    return sorted(uniq)

def load_summary(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_scores(d: Dict, target_activation: float,
                   w_energy: float, w_cap: float, w_stab: float, w_resp: float) -> Tuple[Dict, float]:
    # Pull raw fields with safe defaults
    energy_spent = float(d.get("energy_spent", 0.0))
    energy_recovered = float(d.get("energy_recovered", 0.0))
    time_at_cap_pct = float(d.get("time_at_cap_pct", 0.0))
    oscillation_score = float(d.get("oscillation_score", 0.0))
    act_rate_rep = float(d.get("activation_rate_reported_pct", d.get("activation_rate_pct", 0.0)))

    # Metrics -> 0..1 (higher is better)
    energy_ratio = safe_div(energy_recovered, max(1.0, energy_spent), 0.0)  # could be >1
    energy_eff = clamp01(energy_ratio / 2.0)                                 # cap at ratio 2.0

    stability = 1.0 - clamp01(oscillation_score)                             # assume score in 0..1-ish
    cap_headroom = 1.0 - clamp01(time_at_cap_pct / 100.0)                    # lower cap% is better

    # triangular preference around target_activation
    responsive = 1.0 - clamp01(abs(act_rate_rep - target_activation) / max(1e-6, target_activation))

    # Weighted overall
    total_w = (w_energy + w_cap + w_stab + w_resp) or 1.0
    overall = (
        w_energy * energy_eff +
        w_cap * cap_headroom +
        w_stab * stability +
        w_resp * responsive
    ) / total_w

    scores = {
        "energy_eff": round(energy_eff, 4),
        "cap_headroom": round(cap_headroom, 4),
        "stability": round(stability, 4),
        "responsive": round(responsive, 4),
        "overall": round(overall, 4),
    }
    return scores, overall

def rating_from_overall(x: float) -> str:
    if x >= 0.85: return "A+"
    if x >= 0.75: return "A"
    if x >= 0.65: return "B"
    if x >= 0.55: return "C"
    return "D"

def main():
    ap = argparse.ArgumentParser(description="Build a ranked scorecard from Sundew *.summary.json files")
    ap.add_argument("inputs", nargs="+", help="Files and/or folders. Folders are scanned recursively.")
    ap.add_argument("--pattern", default=DEFAULT_PATTERN, help=f"File pattern when scanning folders (default: {DEFAULT_PATTERN})")
    ap.add_argument("--out", required=True, help="Output JSON path (e.g., data/results/scorecard.json)")
    ap.add_argument("--target-activation", type=float, default=20.0, help="Target activation rate (%%) for responsiveness scoring (default: 20)")
    ap.add_argument("--weights", default="0.35,0.25,0.20,0.20",
                    help="Comma weights for energy,cap,stability,responsive (default: 0.35,0.25,0.20,0.20)")
    args = ap.parse_args()

    try:
        w_energy, w_cap, w_stab, w_resp = [float(x) for x in args.weights.split(",")]
    except Exception:
        ap.error("Invalid --weights. Use four comma-separated numbers, e.g. 0.35,0.25,0.20,0.20")
        return

    files = find_inputs(args.inputs, args.pattern)
    if not files:
        print("[error] No input files matched. Pass files, a folder, or adjust --pattern.")
        return

    items = []
    for path in files:
        try:
            data = load_summary(path)
        except Exception as e:
            print(f"[warn] Skipping {path}: {e}")
            continue

        scores, overall = compute_scores(
            data,
            target_activation=args.target_activation,
            w_energy=w_energy, w_cap=w_cap, w_stab=w_stab, w_resp=w_resp
        )

        name = os.path.basename(path)
        raw_pick = {
            "events": data.get("events"),
            "time_at_cap_pct": data.get("time_at_cap_pct"),
            "oscillation_score": data.get("oscillation_score"),
            "activation_rate_reported_pct": data.get("activation_rate_reported_pct", data.get("activation_rate_pct")),
            "energy_spent": data.get("energy_spent"),
            "energy_recovered": data.get("energy_recovered"),
            "thr_min": data.get("thr_min"),
            "thr_max": data.get("thr_max"),
            "_file": data.get("_file", name),
            "_events_parsed": data.get("_events_parsed", data.get("events")),
        }

        items.append({
            "name": name,
            "path": os.path.relpath(path).replace("\\", "/"),
            "scores": {**scores, "rating": rating_from_overall(scores["overall"])},
            "raw": raw_pick,
        })

    # Rank by overall desc
    items.sort(key=lambda x: x["scores"]["overall"], reverse=True)
    for i, it in enumerate(items, start=1):
        it["rank"] = i

    out_dir = os.path.dirname(os.path.normpath(args.out))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    payload = {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "weights": {
            "energy": w_energy,
            "cap_headroom": w_cap,
            "stability": w_stab,
            "responsive": w_resp,
        },
        "target_activation_pct": args.target_activation,
        "items": items,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {args.out}")
    if items:
        print("Top 3:")
        for it in items[:3]:
            print(f"  {it['rank']:>2}. {it['name']}  overall={it['scores']['overall']:.3f}  rating={it['scores']['rating']}")

if __name__ == "__main__":
    main()
