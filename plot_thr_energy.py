# plot_thr_energy.py
import collections
import csv
from pathlib import Path

import matplotlib.pyplot as plt

CSV = "run_log_video.csv"  # change if needed

if not Path(CSV).exists():
    raise FileNotFoundError(f"Couldn't find {CSV} in {Path.cwd()}")

with open(CSV, newline="") as f:
    rows = list(csv.DictReader(f))

t    = [int(r["frame"]) for r in rows]
thr  = [float(r["threshold"]) for r in rows]
act  = [int(r["activated"]) for r in rows]
eng  = [float(r["energy"]) for r in rows if r.get("energy", "").strip() != ""]

# Threshold with activation dots
plt.figure(figsize=(9, 5.5))
plt.plot(t, thr, linewidth=1)
plt.scatter(
    [tt for tt, a in zip(t, act) if a],
    [th for th, a in zip(thr, act) if a],
    s=6,
)
plt.title("Threshold over time (dots = activations)")
plt.xlabel("frame")
plt.ylabel("threshold")
plt.tight_layout()
plt.savefig("plot_threshold.png", dpi=150)

# Moving activation rate (simple deque-based SMA)
w = 150
rate = []
q: collections.deque[float] = collections.deque()
s = 0.0
for a in act:
    q.append(a)
    s += a
    if len(q) > w:
        s -= q.popleft()
    rate.append(s / len(q))

fig, ax1 = plt.subplots(figsize=(9, 5.5))
ax1.plot(t, thr, label="threshold")
ax1.set_xlabel("frame")
ax1.set_ylabel("threshold")
ax2 = ax1.twinx()
ax2.plot(t, rate, label=f"activation rate (win={w})", alpha=0.85)
ax2.set_ylabel("rate")
plt.title("Threshold vs. moving activation rate")
plt.tight_layout()
plt.savefig("plot_thr_vs_rate.png", dpi=150)

# Energy over time (if present)
if eng:
    plt.figure(figsize=(9, 5.5))
    plt.plot(t[: len(eng)], eng)  # align lengths if needed
    plt.title("Energy over time")
    plt.xlabel("frame")
    plt.ylabel("energy")
    plt.tight_layout()
    plt.savefig("plot_energy.png", dpi=150)

print("Wrote: plot_threshold.png, plot_thr_vs_rate.png", "and plot_energy.png" if eng else "")
