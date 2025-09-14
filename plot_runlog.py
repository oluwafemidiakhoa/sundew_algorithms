# plot_runlog.py
import csv

import matplotlib.pyplot as plt

CSV = r"run_log_video.csv"

rows = list(csv.DictReader(open(CSV, newline="")))
t   = [int(r["frame"]) for r in rows]
thr = [float(r["threshold"]) for r in rows]
eng = [float(r["energy"]) for r in rows if r.get("energy", "").strip() != ""]
act = [int(r["activated"]) for r in rows]

# --- Figure 1: threshold with activation dots ---
plt.figure(figsize=(9, 5.5))
plt.plot(t, thr, linewidth=1)
# overlay activation points
ta = [tt for tt, a in zip(t, act) if a]
tha = [th for th, a in zip(thr, act) if a]
plt.scatter(ta, tha, s=8)
plt.title("Threshold over time (dots = activations)")
plt.xlabel("frame")
plt.ylabel("threshold")
plt.tight_layout()
plt.savefig("plot_threshold.png", dpi=150)

# --- Figure 2: threshold + energy on twin axis ---
plt.figure(figsize=(9, 5.5))
ax1 = plt.gca()
ax1.plot(t, thr, linewidth=1)
ax1.set_xlabel("frame")
ax1.set_ylabel("threshold")

# align energy length if it’s shorter/longer than frames
n = min(len(t), len(eng))
ax2 = ax1.twinx()
ax2.plot(t[:n], eng[:n], linewidth=1)
ax2.set_ylabel("energy")
plt.title("Threshold & Energy over time")
plt.tight_layout()
plt.savefig("plot_threshold_energy.png", dpi=150)

# --- Quick stats ---
activations = sum(act)
rate = activations / len(act) if act else 0.0
print(f"frames={len(t)}, activations={activations}, rate={rate:.3f}")
print("Wrote: plot_threshold.png, plot_threshold_energy.png")
