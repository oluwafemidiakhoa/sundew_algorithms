#!/usr/bin/env python3
# tools/summarize_and_plot.py
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt


# ----------------------------
# Data model and parsing utils
# ----------------------------

@dataclass
class RunRow:
    file: str
    name: str
    rate: float
    activations: int
    total_inputs: int
    precision: float
    recall: float
    f1: float
    savings_pct: float
    energy_left: float
    threshold: float

    @classmethod
    def from_json(cls, path: Path) -> Optional["RunRow"]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Could not read JSON: {path} ({e})", file=sys.stderr)
            return None

        report: Dict[str, Any] = data.get("report", {}) or {}
        counts: Dict[str, Any] = data.get("counts", {}) or {}
        cfg: Dict[str, Any] = data.get("config", {}) or {}

        # Robust fallbacks
        total_inputs = (
            _safe_int(counts.get("total_inputs"))
            or _safe_int(report.get("total_inputs"))
            or 0
        )
        activations = (
            _safe_int(counts.get("activations"))
            or _safe_int(report.get("activations"))
            or 0
        )

        # Activation rate
        rate = _safe_float(report.get("activation_rate"))
        if math.isnan(rate) or rate <= 0:
            rate = (activations / total_inputs) if total_inputs else 0.0

        # Savings %: prefer precomputed; else derive from costs if available
        savings_pct = _safe_float(report.get("estimated_energy_savings_pct"))
        if math.isnan(savings_pct):
            baseline = _safe_float(report.get("baseline_energy_cost"))
            actual = _safe_float(report.get("actual_energy_cost"))
            if baseline > 0 and not math.isnan(actual):
                savings_pct = max(0.0, 100.0 * (1.0 - (actual / baseline)))
            else:
                savings_pct = float("nan")

        precision = _safe_float(counts.get("precision"))
        recall = _safe_float(counts.get("recall"))
        f1 = _safe_float(counts.get("f1"))

        energy_left = _safe_float(report.get("energy_remaining"))
        threshold = _safe_float(report.get("threshold"))

        return cls(
            file=str(path.as_posix()),
            name=path.name,
            rate=rate,
            activations=activations,
            total_inputs=total_inputs,
            precision=precision,
            recall=recall,
            f1=f1,
            savings_pct=savings_pct,
            energy_left=energy_left,
            threshold=threshold,
        )


def _safe_float(x: Any, default=float("nan")) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default=0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default


# ----------------------------
# Discovery & aggregation
# ----------------------------

def discover_jsons(inputs: Sequence[str], glob: str) -> List[Path]:
    files: List[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_file() and p.suffix.lower() == ".json":
            files.append(p.resolve())
        elif p.is_dir():
            files.extend(sorted(p.resolve().rglob(glob)))
        else:
            # treat as glob from CWD
            files.extend(sorted(Path(".").resolve().glob(raw)))
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: List[Path] = []
    for f in files:
        s = f.as_posix()
        if s not in seen:
            seen.add(s)
            unique.append(f)
    return unique


def load_rows(paths: Iterable[Path]) -> List[RunRow]:
    rows: List[RunRow] = []
    for p in paths:
        if p.suffix.lower() != ".json":
            continue
        row = RunRow.from_json(p)
        if row is not None:
            rows.append(row)
    return rows


# ----------------------------
# Sorting, filtering, pareto
# ----------------------------

def sort_rows(rows: List[RunRow], key: str, ascending: bool) -> List[RunRow]:
    key = key.lower()
    def k(r: RunRow):
        if key == "name":
            return r.name
        if key == "rate":
            return r.rate
        if key == "precision":
            return r.precision
        if key == "recall":
            return r.recall
        if key == "f1":
            return r.f1
        if key == "savings" or key == "savings_pct":
            return r.savings_pct
        if key == "threshold":
            return r.threshold
        if key == "energy_left":
            return r.energy_left
        # default: f1
        return r.f1
    return sorted(rows, key=k, reverse=not ascending)


def pareto_front(rows: List[RunRow]) -> List[RunRow]:
    """
    Compute non-dominated set maximizing BOTH f1 and savings_pct.
    A row A dominates B if A.f1 >= B.f1 and A.savings_pct >= B.savings_pct, with one strict.
    """
    pts = list(rows)
    pts.sort(key=lambda r: (r.f1, r.savings_pct), reverse=True)
    front: List[RunRow] = []
    best_savings = -float("inf")
    for r in pts:
        if r.savings_pct > best_savings:
            front.append(r)
            best_savings = r.savings_pct
    return front


# ----------------------------
# Output writers
# ----------------------------

def write_csv(rows: List[RunRow], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "file", "name", "rate", "activations", "total_inputs",
            "precision", "recall", "f1", "savings_pct", "energy_left", "threshold"
        ])
        for r in rows:
            w.writerow([
                r.file, r.name, _fmt(r.rate), r.activations, r.total_inputs,
                _fmt(r.precision), _fmt(r.recall), _fmt(r.f1),
                _fmt(r.savings_pct), _fmt(r.energy_left), _fmt(r.threshold)
            ])


def write_md(rows: List[RunRow], out_md: Path, images: Optional[List[Path]] = None, images_rel_to: Optional[Path] = None) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Results summary\n\n")
        f.write("| file | rate | savings% | P | R | F1 | thr | E_left |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r.name} | {_fmt(r.rate)} | {_fmt(r.savings_pct)}% | "
                f"{_fmt(r.precision)} | {_fmt(r.recall)} | {_fmt(r.f1)} | "
                f"{_fmt(r.threshold)} | {_fmt(r.energy_left)} |\n"
            )
        if images:
            f.write("\n## Plots\n\n")
            for img in images:
                link = img
                if images_rel_to:
                    try:
                        link = img.relative_to(images_rel_to)
                    except ValueError:
                        pass
                f.write(f"![{img.stem}]({link.as_posix()})\n\n")


def write_json(rows: List[RunRow], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(r) for r in rows]
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_console(rows: List[RunRow]) -> None:
    headers = ["file", "rate", "activations", "total_inputs", "precision", "recall", "f1", "savings_pct", "energy_left", "threshold"]
    widths = [max(len(h), *(len(_fmt(getattr(r, "name" if i == 0 else headers[i]))) for r in rows)) for i, h in enumerate(headers)]
    # Fix the first column to 'name' for display
    def row_to_list(r: RunRow) -> List[str]:
        return [
            r.name, _fmt(r.rate), str(r.activations), str(r.total_inputs),
            _fmt(r.precision), _fmt(r.recall), _fmt(r.f1),
            _fmt(r.savings_pct), _fmt(r.energy_left), _fmt(r.threshold)
        ]
    # Headers
    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        vals = row_to_list(r)
        # left-align name, right-align numbers
        line = " | ".join([vals[0].ljust(widths[0])] + [vals[i].rjust(widths[i]) for i in range(1, len(vals))])
        print(line)


def _fmt(x: Any, places: int = 3) -> str:
    if isinstance(x, str):
        return x
    try:
        v = float(x)
    except Exception:
        return str(x)
    if math.isnan(v):
        return "nan"
    if abs(v) >= 1000 or (0 < abs(v) < 1e-3):
        return f"{v:.3e}"
    if v.is_integer():
        return f"{int(v)}"
    return f"{v:.{places}f}"


# ----------------------------
# Plotting
# ----------------------------

def make_plots(rows: List[RunRow], out_dir: Path, annotate_top: int = 8) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    if not rows:
        return paths

    # 1) F1 vs Savings
    fig = plt.figure()
    ax = plt.gca()
    xs = [r.savings_pct for r in rows]
    ys = [r.f1 for r in rows]
    ax.scatter(xs, ys, alpha=0.85)
    ax.set_xlabel("Estimated energy savings (%)")
    ax.set_ylabel("F1")
    ax.set_title("F1 vs. Energy Savings")
    for r in _top_by(rows, key=lambda r: r.f1 * max(r.savings_pct, 0.0), n=annotate_top):
        ax.annotate(r.name.replace(".json", ""), (r.savings_pct, r.f1), xytext=(4, 4), textcoords="offset points", fontsize=8)
    fig.tight_layout()
    p1 = out_dir / "f1_vs_savings.png"
    fig.savefig(p1.as_posix(), dpi=150)
    plt.close(fig)
    paths.append(p1)

    # 2) Precision vs Recall (size ~ savings, marker alpha encodes rate)
    fig = plt.figure()
    ax = plt.gca()
    pr = [(r.precision, r.recall, r) for r in rows]
    sizes = [max(20.0, 2.0 * max(r.savings_pct, 0.0)) for _, _, r in pr]
    alphas = [min(0.9, max(0.3, r.rate * 3.0)) for _, _, r in pr]
    for (p, q, r), s, a in zip(pr, sizes, alphas):
        ax.scatter([p], [q], s=s, alpha=a)
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.set_title("Precision–Recall (size ~ savings%, alpha ~ activation rate)")
    for r in _top_by(rows, key=lambda r: r.f1, n=annotate_top):
        ax.annotate(r.name.replace(".json", ""), (r.precision, r.recall), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    p2 = out_dir / "precision_recall.png"
    fig.savefig(p2.as_posix(), dpi=150)
    plt.close(fig)
    paths.append(p2)

    # 3) Activation rate vs F1
    rows_sorted = sorted(rows, key=lambda r: r.f1, reverse=True)
    fig = plt.figure()
    ax1 = plt.gca()
    idx = list(range(len(rows_sorted)))
    ax1.plot(idx, [r.f1 for r in rows_sorted], marker="o")
    ax1.set_xlabel("runs (sorted by F1)")
    ax1.set_ylabel("F1")
    ax2 = ax1.twinx()
    ax2.plot(idx, [r.rate for r in rows_sorted], linestyle="--")
    ax2.set_ylabel("activation rate")
    ax1.set_title("F1 and activation rate across runs")
    fig.tight_layout()
    p3 = out_dir / "f1_and_rate.png"
    fig.savefig(p3.as_posix(), dpi=150)
    plt.close(fig)
    paths.append(p3)

    # 4) Pareto front (F1↑, Savings↑)
    front = pareto_front(rows)
    if front:
        fig = plt.figure()
        ax = plt.gca()
        ax.scatter(xs, ys, alpha=0.35)
        fx = [r.savings_pct for r in front]
        fy = [r.f1 for r in front]
        # Sort front by savings, connect to visualize frontier
        order = sorted(range(len(front)), key=lambda i: fx[i])
        fx = [fx[i] for i in order]
        fy = [fy[i] for i in order]
        ax.plot(fx, fy, linewidth=2)
        for r in _top_by(front, key=lambda r: r.f1, n=min(annotate_top, len(front))):
            ax.annotate(r.name.replace(".json", ""), (r.savings_pct, r.f1), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_xlabel("Estimated energy savings (%)")
        ax.set_ylabel("F1")
        ax.set_title("Pareto frontier (maximize F1 & savings)")
        fig.tight_layout()
        p4 = out_dir / "pareto_frontier.png"
        fig.savefig(p4.as_posix(), dpi=150)
        plt.close(fig)
        paths.append(p4)

    return paths


def _top_by(rows: List[RunRow], key, n: int) -> List[RunRow]:
    return sorted(rows, key=key, reverse=True)[: max(0, n)]


# ----------------------------
# CLI
# ----------------------------

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Summarize Sundew results JSON and generate CSV/Markdown/plots."
    )
    ap.add_argument(
        "--inputs",
        nargs="*",
        default=["results"],
        help="Files or directories to scan (default: results). Can be mixed.",
    )
    ap.add_argument(
        "--glob",
        default="**/*.json",
        help="Glob to use inside directories (default: **/*.json).",
    )
    ap.add_argument(
        "--sort",
        default="f1",
        help="Sort key: f1|savings|rate|precision|recall|threshold|energy_left|name (default: f1).",
    )
    ap.add_argument(
        "--ascending",
        action="store_true",
        help="Sort ascending (default: descending).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Keep only top-N after sorting (default: all).",
    )
    ap.add_argument(
        "--out-csv",
        default="results/summary.csv",
        help="Output CSV path (default: results/summary.csv).",
    )
    ap.add_argument(
        "--out-md",
        default="results/summary.md",
        help="Output Markdown path (default: results/summary.md).",
    )
    ap.add_argument(
        "--out-json",
        default=None,
        help="Optional JSON export of the summary rows.",
    )
    ap.add_argument(
        "--plots-dir",
        default="results/plots",
        help="Directory to write PNG plots (default: results/plots).",
    )
    ap.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation.",
    )
    ap.add_argument(
        "--annotate-top",
        type=int,
        default=8,
        help="Number of points to annotate in plots (default: 8).",
    )
    ap.add_argument(
        "--md-images-relative-to",
        default=".",
        help="Rewrite image paths in Markdown relative to this directory (default: .).",
    )
    return ap


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    paths = discover_jsons(args.inputs, args.glob)
    if not paths:
        print("[ERROR] No JSON files found. Check --inputs/--glob.", file=sys.stderr)
        sys.exit(2)

    rows = load_rows(paths)
    if not rows:
        print("[ERROR] No valid result rows parsed.", file=sys.stderr)
        sys.exit(2)

    rows = sort_rows(rows, args.sort, args.ascending)
    if args.limit is not None and args.limit > 0:
        rows = rows[: args.limit]

    # Console
    print_console(rows)

    # Outputs
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_json = Path(args.out_json) if args.out_json else None
    plots_dir = Path(args.plots_dir)

    # Plots
    images: List[Path] = []
    if not args.no_plots:
        try:
            images = make_plots(rows, plots_dir, annotate_top=max(0, int(args.annotate_top)))
        except Exception as e:
            print(f"[WARN] Plot generation failed: {e}", file=sys.stderr)

    # CSV / MD / JSON
    try:
        write_csv(rows, out_csv)
        print(f"Wrote CSV: {out_csv.as_posix()}")
    except Exception as e:
        print(f"[ERROR] Writing CSV failed: {e}", file=sys.stderr)

    try:
        base = Path(args.md_images_relative_to).resolve()
        write_md(rows, out_md, images=images, images_rel_to=base)
        print(f"Wrote Markdown: {out_md.as_posix()}")
    except Exception as e:
        print(f"[ERROR] Writing Markdown failed: {e}", file=sys.stderr)

    if out_json:
        try:
            write_json(rows, out_json)
            print(f"Wrote JSON: {out_json.as_posix()}")
        except Exception as e:
            print(f"[ERROR] Writing JSON failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
