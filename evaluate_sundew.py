
#!/usr/bin/env python3
"""
evaluate_sundew.py  (robust version)

- Tolerates missing emojis / encoding differences (✅, ⏸, Δ)
- Parses decision keywords even if emojis are stripped
- Handles empty-event logs gracefully (no crash)
- Computes oscillation metrics only when events exist

Usage:
  python evaluate_sundew.py run1.txt [run2.txt ...]
  type run.txt | python evaluate_sundew.py -
"""

import json
import re
import statistics as stats
import sys
from pathlib import Path
from typing import Any, Dict, List

# Flexible event line:
#  01. environmental   ✅ processed     (sig=0.220, 0.002s, ΔE≈9.5)         | energy   90.5 | thr 0.784
#  03. emergency       processed        (sig=0.783, 0.003s, dE≈11.7)        | energy   79.3 | thr 0.797
EVENT_RE = re.compile(
    r'^\s*(\d+)\.\s+([A-Za-z_]+)\s+(?:\S+\s+)?'         # index, label, optional emoji token
    r'(processed|dormant)\s*'                           # decision keyword (emoji may be stripped)
    r'(?:\(\s*sig=([\d.]+)\s*,\s*([\d.]+)s\s*,\s*(?:ΔE|dE|DE)≈?([\d.]+)\s*\))?'  # optional (sig, latency, dE)
    r'\s*\|\s*energy\s*([-\d.]+)\s*\|\s*thr\s*([-\d.]+)\s*$',
    re.IGNORECASE
)

HEADER_RE = re.compile(r'Initial threshold:\s*([-\d.]+)\s*\|\s*Energy:\s*([-\d.]+)', re.IGNORECASE)
FLOAT_LINE = re.compile(r'^\s*([A-Za-z ]+?):\s*([-+]?[0-9]*\.?[0-9]+)\s*%?\s*$', re.IGNORECASE)

def parse_blocks(text: str) -> Dict[str, Dict[str, float]]:
    sections = {}
    current = None
    for line in text.splitlines():
        if line.strip().endswith(':') and not line.strip().startswith('-'):
            current = line.strip().strip(':')
            sections[current] = {}
            continue
        if current:
            m = FLOAT_LINE.match(line)
            if m:
                key, val = m.group(1).strip(), float(m.group(2))
                sections[current][key] = val
    return sections

def parse(text: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {"events": [], "header": {}, "sections": {}}
    h = HEADER_RE.search(text)
    if h:
        try:
            data["header"]["initial_threshold"] = float(h.group(1))
        except ValueError:
            pass
        try:
            data["header"]["initial_energy"] = float(h.group(2))
        except ValueError:
            pass
    for line in text.splitlines():
        em = EVENT_RE.match(line)
        if em:
            idx = int(em.group(1))
            label = em.group(2)
            decision = em.group(3).lower()
            sig = float(em.group(4)) if em.group(4) else None
            latency = float(em.group(5)) if em.group(5) else None
            dE = float(em.group(6)) if em.group(6) else None
            energy = float(em.group(7))
            thr = float(em.group(8))
            data["events"].append({
                "idx": idx, "label": label, "decision": decision,
                "sig": sig, "latency_s": latency, "dE": dE,
                "energy": energy, "thr": thr
            })
    data["sections"] = parse_blocks(text)
    return data

def derived_metrics(parsed: Dict[str, Any]) -> Dict[str, Any]:
    events = parsed["events"]
    sections = parsed["sections"]
    header = parsed.get("header", {})
    cap = 100.0
    n = len(events)

    processed = sum(1 for e in events if e["decision"] == "processed")
    time_at_cap = sum(1 for e in events if e["energy"] >= cap - 1e-6) if n else 0
    time_at_near_cap = sum(1 for e in events if e["energy"] >= cap - 5.0) if n else 0

    thr_values = [e["thr"] for e in events]
    if len(thr_values) > 1:
        thr_deltas = [thr_values[i] - thr_values[i-1] for i in range(1, len(thr_values))]
        thr_std = float(stats.pstdev(thr_values))
        # zero-crossings of Δthr
        zero_crossings = 0
        for a, b in zip(thr_deltas, thr_deltas[1:]):
            if a == 0 or b == 0:
                continue
            if (a > 0 and b < 0) or (a < 0 and b > 0):
                zero_crossings += 1
        span = (max(thr_values) - min(thr_values)) or 1e-9
        osc_score = (zero_crossings / max(1, len(thr_deltas))) * (thr_std / span)
        plateau_ratio = sum(1 for d in thr_deltas if abs(d) < 1e-9) / max(1, len(thr_deltas))
    else:
        thr_deltas, thr_std, osc_score, plateau_ratio = [], 0.0, 0.0, 0.0

    start_E = header.get("initial_energy", cap)
    sec_energy = sections.get("Energy Analysis", {})
    end_E = sec_energy.get("Energy Remaining", (events[-1]["energy"] if n else start_E))
    recovered = sec_energy.get("Energy Recovered", 0.0)
    spent = sec_energy.get("Total Energy Spent", 0.0)
    recovered_overflow = max(0.0, (start_E + recovered - spent) - end_E)

    latencies = [e.get("latency_s") for e in events if e.get("latency_s") is not None]
    avg_latency = (
        (sum(latencies)/len(latencies)) if latencies
        else sections.get("Control System", {}).get("Avg Processing Time", 0.0)
    )

    return {
        "events": n,
        "processed": processed,
        "activation_rate_pct": round(100.0 * processed / n, 3) if n else None,
        "time_at_cap_pct": round(100.0 * time_at_cap / n, 2) if n else None,
        "time_near_cap_pct": round(100.0 * time_at_near_cap / n, 2) if n else None,
        "thr_min": (min(thr_values) if thr_values else None),
        "thr_max": (max(thr_values) if thr_values else None),
        "thr_std": thr_std,
        "oscillation_score": osc_score,
        "plateau_ratio": plateau_ratio,
        "avg_latency_s": avg_latency,
        "recovered_overflow": recovered_overflow,
        "energy_start": start_E,
        "energy_end": end_E,
        "energy_recovered": recovered,
        "energy_spent": spent,
        "threshold_utilization_pct": (
            sections.get("Control System", {}).get("Threshold Utilization", None)
        ),
        "ema_alpha": sections.get("Control System", {}).get("EMA Alpha", None),
        "ema_rate_reported_pct": (
            sections.get("Performance Metrics", {}).get("EMA Activation Rate", None)
        ),
        "activation_rate_reported_pct": (
            sections.get("Performance Metrics", {}).get("Activation Rate", None)
        ),
        "convergence_status": sections.get("Control System", {}).get("Convergence Status", None),
        "hysteresis_gap": sections.get("Control System", {}).get("Hysteresis Gap", None),
    }

def summarize(path: str, text: str) -> Dict[str, Any]:
    parsed = parse(text)
    d = derived_metrics(parsed)
    d["_file"] = path
    d["_events_parsed"] = len(parsed["events"])
    return d

def print_table(rows: List[Dict[str, Any]]):
    headers = [
        "file", "events(parsed)", "activ% (rep)", "EMA% (rep)", "cap%",
        "near_cap%", "thr_min", "thr_max", "thr_std", "osc_score",
        "plateau", "avg_lat(ms)", "recov_overflow", "spent", "recovered"
    ]
    print("\n=== Sundew Evaluation Summary ===")
    print("\t".join(headers))
    for r in rows:
        def fmt(x):
            if x is None:
                return "-"
            if isinstance(x, float):
                return f"{x:.3f}"
            return str(x)
        print("\t".join([
            Path(r["_file"]).name,
            fmt(r.get("_events_parsed")),
            fmt(r.get("activation_rate_reported_pct")),
            fmt(r.get("ema_rate_reported_pct")),
            fmt(r.get("time_at_cap_pct")),
            fmt(r.get("time_near_cap_pct")),
            fmt(r.get("thr_min")),
            fmt(r.get("thr_max")),
            fmt(r.get("thr_std")),
            fmt(r.get("oscillation_score")),
            fmt(r.get("plateau_ratio")),
            fmt(1000.0 * (r.get("avg_latency_s") or 0.0)),
            fmt(r.get("recovered_overflow")),
            fmt(r.get("energy_spent")),
            fmt(r.get("energy_recovered")),
        ]))
    print("")
    # Warn on files with zero events parsed
    for r in rows:
        if r.get("_events_parsed", 0) == 0:
            print(f"WARNING: {r['_file']} had zero events parsed. "
                  f"Check that you redirected the full CLI output to the file and that lines match the usual format.")

def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)
    rows = []
    for p in args:
        if p == "-":
            text = sys.stdin.read()
            out_path = Path("stdin_log.txt")
        else:
            text = Path(p).read_text(encoding="utf-8", errors="ignore")
            out_path = Path(p)
        r = summarize(str(out_path), text)
        rows.append(r)
        json_out = out_path.with_suffix(out_path.suffix + ".summary.json")
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2)
        print(f"Wrote: {json_out}")
    print_table(rows)

if __name__ == "__main__":
    main()
