#!/usr/bin/env python
"""
Tight Controller Wrapper Demo
-----------------------------
A minimal, self-contained simulation of the "tight" controller:
- Quantile tracking for the threshold to hit a target acceptance rate
- Hysteresis (thr_on/thr_off) to prevent boundary chatter
- Energy-aware target q_eff to conserve when low, relax when full

This does NOT require `sundew` installed; it uses synthetic events to
illustrate behavior and to help you tune eta_q, hyst, and E_min.

Usage:
  python sundew_tight_controller_wrapper.py --steps 500 --target 0.30 \
      --eta 0.02 --hyst 0.02 --reserve 12 --seed 42

Outputs:
  - tight_energy.png
  - tight_threshold.png
  - tight_activations.png
  - tight_cumrate.png

Author: ChatGPT
"""
import argparse
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def gen_event(rng):
    # Synthetic categories with different typical signal ranges
    cat = rng.choices(
        population=["emergency","system_alert","security","health_monitor","environmental"],
        weights=[0.12, 0.22, 0.22, 0.22, 0.22],
        k=1
    )[0]
    base = {
        "emergency": (0.55, 0.20),
        "system_alert": (0.40, 0.12),
        "security": (0.45, 0.12),
        "health_monitor": (0.43, 0.12),
        "environmental": (0.42, 0.12),
    }[cat]
    mu, sigma = base
    sig = float(np.clip(rng.gauss(mu, sigma), 0.0, 1.0))
    return cat, sig

def simulate(steps=500, q_target=0.30, eta_q=0.02, hyst=0.02, E_min=12.0,
             regen_idle=1.7, hard_emergency=0.95, seed=42):
    rng = random.Random(seed)
    thr = 0.78
    thr_on = thr + hyst
    thr_off = thr - hyst
    ema = 0.0
    energy = 100.0
    was_active = False
    m = defaultdict(lambda: 1.0, {"emergency": 1.05})
    rows = []

    for idx in range(1, steps+1):
        cat, sig = gen_event(rng)
        eff_sig = m[cat] * sig
        energy_before = energy
        thr_before = thr

        if was_active:
            should = (eff_sig >= thr_off) or (cat == "emergency" and eff_sig >= hard_emergency)
        else:
            should = (eff_sig >= thr_on) and (energy > E_min)

        if should:
            y = 1.0
            # Slightly cheaper when signal is strong:
            cost = 11.2 - 1.0*(sig - 0.5)
            cost = float(np.clip(cost, 8.5, 12.5))
            energy = max(0.0, energy - cost)
            was_active = True
        else:
            y = 0.0
            energy = float(min(100.0, energy + regen_idle))
            was_active = False

        # Energy-aware target
        q_eff = q_target * (0.6 + 0.4 * (energy/100.0))
        thr = float(np.clip(thr + eta_q*(y - q_eff), 0.05, 0.95))
        thr_on, thr_off = thr + hyst, thr - hyst
        ema = 0.2*y + 0.8*ema

        rows.append({
            "idx": idx, "category": cat, "sig": sig, "eff_sig": eff_sig,
            "y": int(y), "energy": energy, "thr": thr, "thr_on": thr_on, "thr_off": thr_off,
            "ema": ema, "q_eff": q_eff, "thr_before": thr_before, "energy_before": energy_before
        })

    return pd.DataFrame(rows)

def plot(df, prefix="tight"):
    # Energy
    plt.figure()
    plt.plot(df["idx"], df["energy"])
    plt.title("Energy vs Event")
    plt.xlabel("Event index")
    plt.ylabel("Energy")
    plt.savefig(f"{prefix}_energy.png", bbox_inches="tight")
    plt.close()
    # Threshold
    plt.figure()
    plt.plot(df["idx"], df["thr"])
    plt.title("Threshold vs Event")
    plt.xlabel("Event index")
    plt.ylabel("Threshold")
    plt.savefig(f"{prefix}_threshold.png", bbox_inches="tight")
    plt.close()
    # Activations
    plt.figure()
    plt.plot(df["idx"], df["y"])
    plt.title("Activations (1) vs Dormant (0)")
    plt.xlabel("Event index")
    plt.ylabel("Activation flag")
    plt.savefig(f"{prefix}_activations.png", bbox_inches="tight")
    plt.close()
    # Cumulative rate
    cum = df["y"].cumsum() / (df["idx"])
    plt.figure()
    plt.plot(df["idx"], cum)
    plt.title("Cumulative Activation Rate")
    plt.xlabel("Event index")
    plt.ylabel("Cumulative rate")
    plt.savefig(f"{prefix}_cumrate.png", bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--target", type=float, default=0.30)
    ap.add_argument("--eta", type=float, default=0.02)
    ap.add_argument("--hyst", type=float, default=0.02)
    ap.add_argument("--reserve", type=float, default=12.0)
    ap.add_argument("--regen", type=float, default=1.7)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = simulate(steps=args.steps, q_target=args.target, eta_q=args.eta,
                  hyst=args.hyst, E_min=args.reserve, regen_idle=args.regen,
                  seed=args.seed)
    df.to_csv("tight_run.csv", index=False)
    plot(df, "tight")
    print("Done. Files: tight_run.csv, tight_energy.png, tight_threshold.png, "
          "tight_activations.png, tight_cumrate.png")

if __name__ == "__main__":
    main()
