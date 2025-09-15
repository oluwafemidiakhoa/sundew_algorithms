#!/usr/bin/env python
"""
Sundew CLI Tester
-----------------
Run the installed `sundew` demo multiple times, capture stdout logs, parse them,
and produce per-run and aggregate CSVs + plots.

Usage (Windows PowerShell or CMD):
  python sundew_cli_tester.py --runs 5 --events 200 --outdir runs --temperature 0.05

Optional flags:
  --preset <name>           # e.g., conservative | aggressive | ecg_v1
  --temperature <float>     # e.g., 0.05
  --target <float>          # desired activation rate (for reporting only)
  --outdir <folder>         # where results go (default: runs)

The script tries 'sundew' CLI first; if not found, it falls back to:
  py -m sundew.cli   OR   python -m sundew.cli

Outputs per run:
  - run_<i>_raw.txt           (raw CLI output)
  - run_<i>_events.csv        (parsed table: idx, category, status, sig, energy, thr)
  - run_<i>_energy.png        (energy trace)
  - run_<i>_threshold.png     (threshold trace)
  - run_<i>_activations.png   (0/1 activations)
  - run_<i>_signals_thr.png   (sig vs thr for processed steps)

Aggregate:
  - all_runs_summary.csv
  - per_category_summary.csv

Author: ChatGPT
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROW_RE = re.compile(
    r"^\s*(\d+)\.\s*([a-z_]+)\s+(✅ processed|⏸ dormant)(?:\s*\(sig=([0-9.]+),\s*([0-9.]+)s,\s*ΔE≈([0-9.]+)\))?\s*\|\s*energy\s*([0-9.]+)\s*\|\s*thr\s*([0-9.]+)",
    re.MULTILINE
)
INIT_RE = re.compile(r"Initial threshold:\s*([0-9.]+)\s*\|\s*Energy:\s*([0-9.]+)")

def call_sundew(events: int, preset: str|None, temperature: float|None) -> str:
    """Run the sundew CLI and return stdout as text."""
    args = []
    # Preferred command
    cmd_candidates = [
        ["sundew"],
        ["py", "-m", "sundew.cli"],
        ["python", "-m", "sundew.cli"],
        ["python3", "-m", "sundew.cli"],
    ]
    base_args = ["--demo", "--events", str(events)]
    if preset:
        base_args += ["--preset", preset]
    if temperature is not None:
        base_args += ["--temperature", str(temperature)]
    last_err = None
    for cmd in cmd_candidates:
        try:
            cp = subprocess.run(cmd + base_args, capture_output=True, text=True, check=True)
            return cp.stdout
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not invoke sundew CLI. Last error: {last_err}")

def parse_run_text(text: str) -> pd.DataFrame:
    """Parse CLI text into a DataFrame of events."""
    rows = []
    for m in ROW_RE.finditer(text):
        idx = int(m.group(1))
        category = m.group(2)
        status = "processed" if "processed" in m.group(3) else "dormant"
        sig = float(m.group(4)) if m.group(4) else np.nan
        dur = float(m.group(5)) if m.group(5) else np.nan
        dE  = float(m.group(6)) if m.group(6) else np.nan
        energy = float(m.group(7))
        thr = float(m.group(8))
        rows.append((idx, category, status, sig, dur, dE, energy, thr))
    df = pd.DataFrame(rows, columns=["idx", "category", "status", "sig", "duration_s", "delta_E", "energy", "thr"])
    df["is_processed"] = (df["status"] == "processed").astype(int)
    return df

def save_plots(out_dir: Path, df: pd.DataFrame, run_name: str):
    # Energy
    plt.figure()
    plt.plot(df["idx"], df["energy"])
    plt.title(f"Energy vs Event — {run_name}")
    plt.xlabel("Event index")
    plt.ylabel("Energy")
    plt.savefig(out_dir / f"{run_name}_energy.png", bbox_inches="tight")
    plt.close()

    # Threshold
    plt.figure()
    plt.plot(df["idx"], df["thr"])
    plt.title(f"Threshold vs Event — {run_name}")
    plt.xlabel("Event index")
    plt.ylabel("Threshold")
    plt.savefig(out_dir / f"{run_name}_threshold.png", bbox_inches="tight")
    plt.close()

    # Activations
    plt.figure()
    plt.plot(df["idx"], df["is_processed"])
    plt.title(f"Activations (1) vs Dormant (0) — {run_name}")
    plt.xlabel("Event index")
    plt.ylabel("Activation flag")
    plt.savefig(out_dir / f"{run_name}_activations.png", bbox_inches="tight")
    plt.close()

    # Signals vs Threshold
    proc_df = df[df["is_processed"] == 1].copy()
    plt.figure()
    if not proc_df.empty:
        plt.scatter(proc_df["idx"], proc_df["sig"])
    plt.plot(df["idx"], df["thr"])
    plt.title(f"Signals (processed) & Threshold — {run_name}")
    plt.xlabel("Event index")
    plt.ylabel("Signal / Threshold")
    plt.savefig(out_dir / f"{run_name}_signals_thr.png", bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--events", type=int, default=200)
    ap.add_argument("--preset", type=str, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--target", type=float, default=0.30)
    ap.add_argument("--outdir", type=str, default="runs")
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    per_cat_rows = []
    for i in range(1, args.runs+1):
        run_name = f"run_{i:02d}"
        print(f"[+] Running demo {i}/{args.runs} ...", flush=True)
        try:
            text = call_sundew(args.events, args.preset, args.temperature)
        except Exception as e:
            print(f"[!] Failed to run sundew: {e}")
            sys.exit(1)

        (out_dir / f"{run_name}_raw.txt").write_text(text, encoding="utf-8")

        df = parse_run_text(text)
        if df.empty:
            print("[!] No rows parsed — check CLI output format.")
            sys.exit(2)
        df.to_csv(out_dir / f"{run_name}_events.csv", index=False)
        save_plots(out_dir, df, run_name)

        # Per-run summary
        act = int(df["is_processed"].sum())
        rate = float(df["is_processed"].mean())
        energy_end = float(df["energy"].iloc[-1])
        thr_end = float(df["thr"].iloc[-1])
        summaries.append({
            "run": run_name,
            "events": int(df["idx"].count()),
            "activations": act,
            "activation_rate": rate,
            "energy_remaining": energy_end,
            "threshold_end": thr_end,
            "target_activation_rate": args.target
        })

        # Per-category
        pc = df.groupby("category").agg(events=("idx","count"),
                                        processed=("is_processed","sum"))
        pc["rate"] = pc["processed"] / pc["events"]
        for cat, row in pc.iterrows():
            per_cat_rows.append({
                "run": run_name,
                "category": cat,
                "events": int(row["events"]),
                "processed": int(row["processed"]),
                "rate": float(row["rate"]),
            })

    pd.DataFrame(summaries).to_csv(out_dir / "all_runs_summary.csv", index=False)
    pd.DataFrame(per_cat_rows).to_csv(out_dir / "per_category_summary.csv", index=False)
    print(f"[✓] Done. See folder: {out_dir}")

if __name__ == "__main__":
    main()
