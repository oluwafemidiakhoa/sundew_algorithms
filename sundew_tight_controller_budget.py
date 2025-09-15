#!/usr/bin/env python
"""
Tight Controller (energy-budget aware)
-------------------------------------
Adds a sustainable-rate clip so the target acceptance never exceeds
what your energy budget can support:  r_sustain ≈ regen_idle / cost_avg

Usage:
  python sundew_tight_controller_budget.py --steps 500 --target 0.30 \
     --eta 0.04 --hyst 0.03 --reserve 20 --regen 2.2 --seed 42
"""
import argparse
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def gen_event(rng):
    cat = rng.choices(
        population=["emergency","system_alert","security","health_monitor","environmental"],
        weights=[0.12, 0.22, 0.22, 0.22, 0.22],
        k=1
    )[0]
    mu, sigma = {"emergency":(0.58,0.20), "system_alert":(0.42,0.12),
                 "security":(0.47,0.12), "health_monitor":(0.45,0.12),
                 "environmental":(0.44,0.12)}[cat]
    sig = float(np.clip(rng.gauss(mu, sigma), 0.0, 1.0))
    return cat, sig

def simulate(steps=500, q_target=0.30, eta_q=0.04, hyst=0.03, E_min=20.0,
             regen_idle=2.2, hard_emergency=0.95, seed=42):
    rng = random.Random(seed)
    thr = 0.78; thr_on = thr+hyst; thr_off = thr-hyst
    ema = 0.0; energy = 100.0; was_active = False
    m = defaultdict(lambda: 1.0, {"emergency": 1.05})
    cost_avg = 11.0  # initial guess
    beta_cost = 0.1  # EMA for cost

    rows = []
    for idx in range(1, steps+1):
        cat, sig = gen_event(rng)
        eff_sig = m[cat]*sig
        energy_before, thr_before = energy, thr

        if was_active:
            should = (eff_sig >= thr_off) or (cat=="emergency" and eff_sig>=hard_emergency)
        else:
            should = (eff_sig >= thr_on) and (energy > E_min)

        if should:
            y = 1.0
            # Slightly cheaper with strong signals
            cost = 11.2 - 1.0*(sig - 0.5)
            cost = float(np.clip(cost, 8.5, 12.5))
            energy = max(0.0, energy - cost)
            cost_avg = (1-beta_cost)*cost_avg + beta_cost*cost
            was_active = True
        else:
            y = 0.0
            energy = float(min(100.0, energy + regen_idle))
            was_active = False

        # Sustainable rate estimate and energy-aware cap
        r_sustain = min(0.9, regen_idle / max(cost_avg, 1e-6))  # avoid >1.0
        cap = 0.85 * r_sustain
        q_energy = (0.4 + 0.6*(energy/100.0)) * q_target
        q_eff = min(q_energy, cap)

        thr = float(np.clip(thr + eta_q*(y - q_eff), 0.05, 0.95))
        thr_on, thr_off = thr + hyst, thr - hyst
        ema = 0.2*y + 0.8*ema

        rows.append({
            "idx": idx, "category": cat, "sig": sig, "eff_sig": eff_sig,
            "y": int(y), "energy": energy, "thr": thr, "thr_on": thr_on, "thr_off": thr_off,
            "ema": ema, "q_eff": q_eff, "r_sustain": r_sustain,
            "thr_before": thr_before, "energy_before": energy_before, "cost_avg": cost_avg
        })
    return pd.DataFrame(rows)

def plot(df, prefix="tight_budget"):
    # Energy
    plt.figure(); plt.plot(df["idx"], df["energy"])
    plt.title("Energy vs Event"); plt.xlabel("Event index"); plt.ylabel("Energy")
    plt.savefig(f"{prefix}_energy.png", bbox_inches="tight"); plt.close()
    # Threshold
    plt.figure(); plt.plot(df["idx"], df["thr"])
    plt.title("Threshold vs Event"); plt.xlabel("Event index"); plt.ylabel("Threshold")
    plt.savefig(f"{prefix}_threshold.png", bbox_inches="tight"); plt.close()
    # Activations
    plt.figure(); plt.plot(df["idx"], df["y"])
    plt.title("Activations (1) vs Dormant (0)"); plt.xlabel("Event index"); plt.ylabel("Activation flag")
    plt.savefig(f"{prefix}_activations.png", bbox_inches="tight"); plt.close()
    # Cumulative rate
    cum = df["y"].cumsum()/df["idx"]
    plt.figure(); plt.plot(df["idx"], cum)
    plt.title("Cumulative Activation Rate"); plt.xlabel("Event index"); plt.ylabel("Cumulative rate")
    plt.savefig(f"{prefix}_cumrate.png", bbox_inches="tight"); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--target", type=float, default=0.30)
    ap.add_argument("--eta", type=float, default=0.04)
    ap.add_argument("--hyst", type=float, default=0.03)
    ap.add_argument("--reserve", type=float, default=20.0)
    ap.add_argument("--regen", type=float, default=2.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    df = simulate(steps=args.steps, q_target=args.target, eta_q=args.eta,
                  hyst=args.hyst, E_min=args.reserve, regen_idle=args.regen,
                  seed=args.seed)
    df.to_csv("tight_budget_run.csv", index=False)
    plot(df, "tight_budget")
    # Print quick stats
    rate = df["y"].mean()
    print(f"Final activation rate: {rate:.3f}")
    print(f"Avg sustainable estimate r*: {df['r_sustain'].mean():.3f}")
    print("Files: tight_budget_run.csv, tight_budget_*png")

if __name__ == "__main__":
    main()
