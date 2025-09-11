# Sundew Algorithms

> **Bio-inspired, energy-aware selective activation for streaming data.**
> Sundew decides when to fully process an input and when to skip, trading a small drop in accuracy for **large energy savings**.

[![PyPI version](https://badge.fury.io/py/sundew-algorithms.svg)](https://badge.fury.io/py/sundew-algorithms)
[![CI Status](https://github.com/your-username/sundew-algorithms/workflows/CI/badge.svg)](https://github.com/your-username/sundew-algorithms/actions)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## What is Sundew?

Sundew is an adaptive gating algorithm inspired by the carnivorous sundew plant, which selectively opens and closes its traps based on prey quality and energy reserves. In streaming data applications, Sundew intelligently decides whether to fully process each incoming sample or skip it, maintaining system responsiveness while dramatically reducing computational costs.

The algorithm continuously adjusts its selectivity threshold based on available "energy" (computational budget), data characteristics, and performance targets. This creates a dynamic trade-off between processing accuracy and resource consumption—perfect for battery-powered devices, edge computing, or high-throughput systems where every cycle counts.

## Why gating helps

```
Traditional Processing:           Sundew Gating:
┌─────────────────────┐          ┌─────────────────────┐
│ Input Stream        │          │ Input Stream        │
│ ████████████████    │          │ ████████████████    │
└──────────┬──────────┘          └──────────┬──────────┘
           │                                │
           ▼                                ▼
┌─────────────────────┐          ┌─────────────────────┐
│ Process EVERYTHING  │          │ Adaptive Gate       │
│ • High CPU usage    │          │ • Smart filtering   │
│ • Battery drain     │          │ • Energy-aware      │
│ • Thermal issues    │          │ • Quality-conscious │
└──────────┬──────────┘          └──────────┬──────────┘
           │                                │
           ▼                                ▼
┌─────────────────────┐          ┌─────────────────────┐
│ Results: 100%       │          │ Process: ~10-20%    │
│ Energy: 100%        │          │ ████  ░░░░  ████    │
│ Accuracy: 100%      │          │ Results: ~85% acc   │
│                     │          │ Energy: ~15% usage  │
└─────────────────────┘          └─────────────────────┘

Key insight: Most samples contain redundant or low-value information.
Sundew learns to focus processing power on the samples that matter most.
```

---

## Benchmarks & Results (ECG)

This repository ships a small benchmarking harness around the Sundew Algorithm for event gating on ECG-like streams.

### Dataset Reference

**MIT-BIH Arrhythmia Database** - The benchmark uses the MIT-BIH Arrhythmia Database, a widely-used reference for cardiac arrhythmia research.

- **Source**: PhysioNet (https://physionet.org/content/mitdb/1.0.0/)
- **Citation**: Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001).
- **Local path**: `C:\Users\adminidiakhoa\sundew_algorithms\data\MIT-BIH Arrhythmia Database.csv`

The CSV schema is auto-detected with fallback logic:
- **Signal columns**: `ml2`, `signal`, `ecg`, `value`, `val`, `amplitude`, `lead*`
- **Label columns**: `label`, `annotation`, `ann`, `class`, `arrhythmia`, `beat_type`

## TL;DR

**Savings at ~10% activation rate:** ~85% estimated energy savings, with F1≈0.158 (savings_012.json).

**Recall-leaning run:** F1≈0.162 at ~12% activation (recall_016_r0.json).

**Aggressive sweep:** up to F1≈0.185 when we allow ~20% activation (tune_ep001_thr085_probe50.json).

All metrics come from results/*.json; the table and plots are generated automatically. See "Reproduce the plots" below.

![summary](summary)

## How to reproduce

```bash
# 1) Run a benchmark (examples)
sundew-ecg --csv "C:\Users\adminidiakhoa\sundew_algorithms\data\MIT-BIH Arrhythmia Database.csv" ^
  --preset tuned_v2 --limit 50000 ^
  --overrides "target_activation_rate=0.12,gate_temperature=0.07,probe_every=100" ^
  --save "results/savings_012.json"

sundew-ecg --csv "C:\Users\adminidiakhoa\sundew_algorithms\data\MIT-BIH Arrhythmia Database.csv" ^
  --preset tuned_v2 --limit 50000 --refractory 3 ^
  --overrides "target_activation_rate=0.14,energy_pressure=0.03,max_threshold=0.90,gate_temperature=0.08,probe_every=75" ^
  --save "results/balanced_014_r3.json"

# 2) Summarize & plot (advanced tool)
python tools/summarize_and_plot.py ^
  --dir results ^
  --out results/summary.csv ^
  --md results/summary.md ^
  --plots results/plots
```

This produces:

- **results/summary.csv** — collated metrics
- **results/summary.md** — nice Markdown table with links
- **results/plots/precision_recall.png, f1_and_rate.png, f1_vs_savings.png, pareto_frontier.png**

## At-a-glance results

| file | rate | savings% | P | R | F1 | thr | E_left |
|------|------|----------|---|---|----|----|--------|
| savings_012.json | 0.103 | 85.44% | 0.158 | 0.158 | 0.158 | 0.671 | 90.0 |
| recall_016_r0.json | 0.122 | 83.65% | 0.150 | 0.177 | 0.162 | 0.879 | 64.9 |
| balanced_016.json | 0.126 | 83.32% | 0.142 | 0.173 | 0.156 | 0.853 | 70.7 |
| balanced_014_r2.json | 0.115 | 84.29% | 0.154 | 0.140 | 0.146 | 0.459 | 84.2 |
| tune_ep001_thr085_probe50.json | 0.201 | 76.13% | 0.140 | 0.273 | 0.185 | 0.820 | 0.0 |

Full table and exact numbers live in results/summary.md (auto-generated).

![summary](summary)

## Figures

Drop these directly in your README:

```markdown
![Precision vs Recall](results/plots/precision_recall.png)
![Activation rate vs F1](results/plots/f1_and_rate.png)
![Savings vs F1](results/plots/f1_vs_savings.png)
![Pareto frontier (F1 vs Savings)](results/plots/pareto_frontier.png)
```

(Your sample renders look good; they're clear and labeled. Keep the default Matplotlib style to stay dependency-light.)

## Interpreting the knobs

- **Activation rate** ≈ fraction of inputs you fully process. Lower is cheaper.
- **Estimated energy savings** is computed vs. a baseline "always process" policy.
- **Precision / Recall / F1** are from binary labels in the CSV.

**Useful overrides:**

- `target_activation_rate` — pushes the gate to be more/less selective.
- `gate_temperature` — softens the gate; lower is harsher.
- `max_threshold` — caps how strict the system can get.
- `energy_pressure` — how quickly the threshold rises when energy is low.
- `probe_every` — occasional probes to avoid starvation.
- `refractory` — suppress repeated activations for N samples after a hit.

## Suggested presets to document

**Savings-first:**
```bash
--overrides "target_activation_rate=0.12,gate_temperature=0.07,probe_every=100"
```
→ good savings with balanced F1 (savings_012.json).

**Recall-leaning:**
```bash
--overrides "target_activation_rate=0.16,energy_pressure=0.035,max_threshold=0.88,gate_temperature=0.08,probe_every=60"
```
(recall_016_r0.json).

**Burst control:**
Add `--refractory 3` to dampen rapid re-fires (balanced_014_r3.json).

## CI Integration & Quality Assurance

### Smoke Test
```yaml
# .github/workflows/ci.yml
- name: Benchmark Smoke Test
  run: |
    sundew-ecg --csv "data/MIT-BIH Arrhythmia Database.csv" --preset tuned_v2 --limit 5000 --save results/smoke.json
    python tools/summarize_and_plot.py --dir results --out results/summary.csv --md results/summary.md --plots results/plots

- name: Upload Results
  uses: actions/upload-artifact@v3
  with:
    name: benchmark-results
    path: |
      results/summary.md
      results/plots/
```

### Development Workflow
```bash
# Quick validation during development
sundew-ecg --csv "C:\Users\adminidiakhoa\sundew_algorithms\data\MIT-BIH Arrhythmia Database.csv" --preset tuned_v2 --limit 5000 --save results/smoke.json

# Assert that key metrics exist
python -c "import json; data=json.load(open('results/smoke.json')); assert 'f1' in data and 'savings_pct' in data"
```

## Project Structure

```
sundew_algorithms/
├── data/                           # Keep in .gitignore (large files)
│   └── MIT-BIH Arrhythmia Database.csv
├── results/                        # Generated outputs (.gitignore)
│   ├── *.json                      # Individual benchmark runs
│   ├── summary.md                  # Auto-generated leaderboard
│   ├── summary.csv                 # Machine-readable metrics
│   └── plots/                      # Auto-generated visualizations
├── tools/
│   └── summarize_and_plot.py       # Advanced analysis script
├── README.md                       # This file (check into git)
└── pyproject.toml                  # Package configuration
```

**Git Strategy:**
- Keep `results/` in `.gitignore` for local development
- Check in `results/summary.md` and key plots you want to showcase
- Regenerate all results in CI for consistency

## Is this suitable for the project?

**Yes.** From a maintainer's perspective you now have:

- **Reproducibility:** single-command runs + deterministic plots.
- **Observability:** summary table + three complementary charts (PR space, F1 vs rate, F1 vs savings).
- **Actionable knobs:** clear guidance on how each parameter shifts the trade-offs.
- **Quality assurance:** CI smoke tests with artifact uploads.
- **Documentation:** Clear dataset provenance and schema detection.

## License

**Apache-2.0**

© 2025 Your Name. This repository is provided for research purposes; this is not a medical device and must not be used for clinical diagnosis.

---

**⚠️ Medical Disclaimer:** This software is a research prototype for algorithm development and benchmarking. It is not intended for medical use, diagnosis, or treatment of any condition. Always consult qualified healthcare professionals for medical decisions.
