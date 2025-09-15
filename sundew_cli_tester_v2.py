#!/usr/bin/env python
"""
Sundew CLI Tester (v2, robust parser)
-------------------------------------
- More tolerant regex: ignores emojis/extra tokens, works with different spacing.
- Optional: parse from a JSON file you saved with `--save` instead of stdout.
- Lets you explicitly choose the command via --cmd.

Examples:
  python sundew_cli_tester_v2.py --runs 3 --events 200 --outdir runs
  python sundew_cli_tester_v2.py --runs 1 --events 100 --cmd "sundew" --outdir runs
  python sundew_cli_tester_v2.py --from_json results.json --outdir runs

Tip: If your CLI supports it, run:
  sundew --demo --events 200 --save results.json
Then parse:
  python sundew_cli_tester_v2.py --from_json results.json
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Robust line parser: minimal assumptions
LINE_RE = re.compile(
    r"^\s*(?P<idx>\d+)\.\s*(?P<cat>[A-Za-z_]+)\s+.*?\b(?P<status>processed|dormant)\b.*?\|\s*energy\s*(?P<energy>[\d.]+)\s*\|\s*thr\s*(?P<thr>[\d.]+)",
    re.IGNORECASE
)

SIG_RE = re.compile(r"sig\s*=\s*([\d.]+)")

def parse_text(raw: str) -> pd.DataFrame:
    rows = []
    for line in raw.splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        idx = int(m.group('idx'))
        cat = m.group('cat')
        status = m.group('status').lower()
        energy = float(m.group('energy'))
        thr = float(m.group('thr'))
        sig_m = SIG_RE.search(line)
        sig = float(sig_m.group(1)) if sig_m else np.nan
        rows.append((idx, cat, status, sig, energy, thr))
    df = pd.DataFrame(rows, columns=["idx","category","status","sig","energy","thr"])
    if df.empty:
        return df
    df["is_processed"] = (df["status"]=="processed").astype(int)
    return df

def parse_json(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    # Expect either {'events': [...]} or a top-level list of events; fallback safely.
    events = data.get("events", data if isinstance(data, list) else [])
    rows = []
    for e in events:
        idx = e.get("index") or e.get("idx") or e.get("i")
        cat = e.get("category", "unknown")
        status = "processed" if e.get("processed") or e.get("status")=="processed" else "dormant"
        sig = e.get("signal") or e.get("sig") or np.nan
        energy = e.get("energy_after") or e.get("energy") or np.nan
        thr = e.get("threshold") or e.get("thr") or np.nan
        if idx is None or energy is None or thr is None:
            continue
        rows.append((int(idx), cat, status, float(sig) if sig is not None else np.nan, float(energy), float(thr)))
    df = pd.DataFrame(rows, columns=["idx","category","status","sig","energy","thr"])
    if df.empty:
        return df
    df["is_processed"] = (df["status"]=="processed").astype(int)
    return df

def save_plots(out_dir: Path, df: pd.DataFrame, name: str):
    # Energy
    plt.figure()
    plt.plot(df["idx"], df["energy"])
    plt.title(f"Energy vs Event — {name}")
    plt.xlabel("Event index"); plt.ylabel("Energy")
    plt.savefig(out_dir / f"{name}_energy.png", bbox_inches="tight")
    plt.close()
    # Threshold
    plt.figure()
    plt.plot(df["idx"], df["thr"])
    plt.title(f"Threshold vs Event — {name}")
    plt.xlabel("Event index"); plt.ylabel("Threshold")
    plt.savefig(out_dir / f"{name}_threshold.png", bbox_inches="tight")
    plt.close()
    # Activations
    plt.figure()
    plt.plot(df["idx"], df["is_processed"])
    plt.title(f"Activations (1) vs Dormant (0) — {name}")
    plt.xlabel("Event index"); plt.ylabel("Activation flag")
    plt.savefig(out_dir / f"{name}_activations.png", bbox_inches="tight")
    plt.close()

    # Cum rate
    cum = df["is_processed"].cumsum()/df["idx"]
    plt.figure()
    plt.plot(df["idx"], cum)
    plt.title(f"Cumulative Activation Rate — {name}")
    plt.xlabel("Event index"); plt.ylabel("Cumulative rate")
    plt.savefig(out_dir / f"{name}_cumrate.png", bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--events", type=int, default=200)
    ap.add_argument("--cmd", type=str, default=None, help="Command to run sundew, e.g., 'sundew' or 'py -m sundew.cli'")
    ap.add_argument("--from_json", type=str, default=None, help="Parse results from a JSON file instead of stdout")
    ap.add_argument("--outdir", type=str, default="runs_v2")
    args = ap.parse_args()

    out_dir = Path(args.outdir); out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_json:
        df = parse_json(Path(args.from_json))
        if df.empty:
            print("[!] JSON parse produced no rows. Check the file format.")
            sys.exit(2)
        df.to_csv(out_dir/"json_events.csv", index=False)
        save_plots(out_dir, df, "json")
        print(f"[✓] Parsed JSON. See {out_dir}")
        return

    cmd = args.cmd or "sundew"
    for i in range(1, args.runs+1):
        name = f"run_{i:02d}"
        print(f"[+] Running: {cmd} --demo --events {args.events}", flush=True)
        cp = subprocess.run(f'{cmd} --demo --events {args.events}', shell=True, capture_output=True, text=True)
        raw = cp.stdout or cp.stderr
        (out_dir/f"{name}_raw.txt").write_text(raw, encoding="utf-8")
        df = parse_text(raw)
        if df.empty:
            print(f"[!] No rows parsed for {name}. Saved raw text at {out_dir}/{name}_raw.txt")
            continue
        df.to_csv(out_dir/f"{name}_events.csv", index=False)
        save_plots(out_dir, df, name)
        print(f"[✓] {name}: events={len(df)}, activations={int(df['is_processed'].sum())}, rate={df['is_processed'].mean():.3f}")
    print("[✓] Done.")

if __name__ == "__main__":
    main()
