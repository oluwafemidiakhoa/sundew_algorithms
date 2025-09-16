#!/usr/bin/env python3
# summarize_results.py
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

Cols = Tuple[str, ...]
DEFAULT_COLS: Cols = (
    "file",
    "rate",
    "activations",
    "total_inputs",
    "precision",
    "recall",
    "f1",
    "savings_pct",
    "energy_left",
    "threshold",
)

def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _fmt(x: Optional[float], nd: int = 3) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "-"

def collect(dirpath: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in sorted(glob.glob(os.path.join(dirpath, "*.json"))):
        try:
            d = json.load(open(p, "r", encoding="utf-8"))
        except Exception:
            continue
        d.get("report", {})
        cnt = d.get("counts", {})

        rows.append({
            "file": os.path.basename(p),
            "rate": _get(d, "report.activation_rate"),
            "activations": _get(d, "report.activations"),
            "total_inputs": _get(d, "report.total_inputs"),
            "precision": cnt.get("precision"),
            "recall": cnt.get("recall"),
            "f1": cnt.get("f1"),
            "savings_pct": _get(d, "report.estimated_energy_savings_pct"),
            "energy_left": _get(d, "report.energy_remaining"),
            "threshold": _get(d, "report.threshold"),
        })
    return rows

def to_markdown(rows: List[Dict[str, Any]], cols: Cols = DEFAULT_COLS) -> str:
    head = "| " + " | ".join(cols) + " |"
    sep  = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [head, sep]
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c)
            if c in {"rate", "precision", "recall", "f1"}:
                vals.append(_fmt(v, 3))
            elif c in {"savings_pct", "energy_left", "threshold"}:
                # savings often looks like a percent already; show 2dp
                vals.append(_fmt(v, 2))
            else:
                vals.append(str(v) if v is not None else "-")
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)

def to_csv(rows: List[Dict[str, Any]], cols: Cols = DEFAULT_COLS) -> str:
    out = [",".join(cols)]
    for r in rows:
        vals: List[str] = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, (int, float)):
                vals.append(str(v))
            elif v is None:
                vals.append("")
            else:
                # naive CSV escaping
                s = str(v).replace('"', '""')
                if "," in s or " " in s:
                    s = f"\"{s}\""
                vals.append(s)
        out.append(",".join(vals))
    return "\n".join(out)

def sort_rows(rows: List[Dict[str, Any]], key: Optional[str]) -> List[Dict[str, Any]]:
    if not key:
        return rows
    def kf(r: Dict[str, Any]):
        v = r.get(key)
        return (v is None, v)  # None → bottom
    return sorted(rows, key=kf, reverse=True)

def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize Sundew JSON results.")
    ap.add_argument("--dir", default="results", help="Directory to scan (default: results)")
    ap.add_argument("--out", default=None, help="Optional CSV path to save")
    ap.add_argument("--md", default=None, help="Optional Markdown path to save")
    ap.add_argument("--sort", default="f1", help="Sort by column (e.g., f1, savings_pct, rate)")
    args = ap.parse_args()

    rows = collect(args.dir)
    rows = sort_rows(rows, args.sort)

    # Console table (simple fixed-width)
    cols = DEFAULT_COLS
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    line = " | ".join(c.ljust(widths[c]) for c in cols)
    print(line)
    print("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        def show(c: str) -> str:
            v = r.get(c)
            if c in {"rate", "precision", "recall", "f1"}:
                return _fmt(v, 3).rjust(widths[c])
            if c in {"savings_pct", "energy_left", "threshold"}:
                return _fmt(v, 2).rjust(widths[c])
            return str(v if v is not None else "").ljust(widths[c])
        print(" | ".join(show(c) for c in cols))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(to_csv(rows, cols))
        print(f"\nWrote CSV: {args.out}")
    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(to_markdown(rows, cols))
        print(f"Wrote Markdown: {args.md}")

if __name__ == "__main__":
    main()
