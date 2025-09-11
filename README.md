# Sundew Algorithms

> **Bio-inspired, energy-aware selective activation for streaming data.**
> Sundew decides when to fully process an input and when to skip, trading a tiny drop in accuracy for very large energy savings—ideal for edge devices, wearables, and high-throughput pipelines.

[![PyPI version](https://badge.fury.io/py/sundew-algorithms.svg)](https://badge.fury.io/py/sundew-algorithms)
[![CI Status](https://github.com/your-username/sundew-algorithms/workflows/CI/badge.svg)](https://github.com/your-username/sundew-algorithms/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Contents

- [Quick start](#quick-start)
- [Why gating helps](#why-gating-helps)
- [Minimal API example](#minimal-api-example)
- [CLI demo](#cli-demo)
- [ECG benchmark (reproduce numbers & plots)](#ecg-benchmark-reproduce-numbers--plots)
- [API cheatsheet](#api-cheatsheet)
- [Configuration presets](#configuration-presets)
- [Results you can paste in blogs/papers](#results-you-can-paste-in-blogspapers)
- [Project structure](#project-structure)
- [License & disclaimer](#license--disclaimer)

---

## Quick start

```bash
# Latest release
pip install -U sundew-algorithms

# Check it installed (Windows examples)
py -3.13 -m sundew --help
py -3.13 -c "import importlib.metadata as m, sundew, sys; print(sundew.__file__); print(m.version('sundew-algorithms')); print(sys.executable)"
```

## Why gating helps

**Traditional: process EVERYTHING**
- High compute, heat, battery drain

**Sundew: process ONLY the valuable ~10–30%**
- Learns a threshold from stream statistics & energy
- Keeps accuracy competitive while slashing energy cost

## Minimal API example

```python
# minimal_api.py
from sundew import SundewAlgorithm
from sundew.config import SundewConfig

cfg = SundewConfig(  # tuned for balanced savings/recall
    activation_threshold=0.78,
    target_activation_rate=0.15,
    gate_temperature=0.08,
    max_threshold=0.92,
    energy_pressure=0.04,
)

algo = SundewAlgorithm(cfg)

# Dummy input with signal features Sundew understands
x = {"magnitude": 63, "anomaly_score": 0.52, "context_relevance": 0.31, "urgency": 0.18}

res = algo.process(x)
if res:
    print(f"Activated: significance={res.significance:.3f}, energy={res.energy_consumed:.2f}")
else:
    print("Skipped (gate dormant)")

# Summary after any loop:
print(algo.report())
```

**Run:**
```bash
py -3.13 minimal_api.py
```

## CLI demo

Interactive demo with emojis and a final report:

```bash
py -3.13 -m sundew --demo --events 50 --temperature 0.08 --save "%USERPROFILE%\Downloads\demo_run.json"
```

You'll see lines like:
```
07. health_monitor  ✅ processed (sig=0.710, 0.003s, ΔE≈11.4) | energy   73.5 | thr 0.816
…
🏁 Final Report
  activation_rate               : 0.160
  energy_remaining              : 66.659
  estimated_energy_savings_pct  : 80.04%
```

**Small helper to summarize that JSON:**
```bash
py -3.13 tools\summarize_demo_json.py
```

**And a quick histogram of processed event significances:**
```bash
pip install matplotlib
py -3.13 tools\plot_significance_hist.py --json "%USERPROFILE%\Downloads\demo_run.json" --bins 24
```

## ECG benchmark (reproduce numbers & plots)

We include a simple CSV benchmark for the MIT-BIH Arrhythmia dataset (CSV export). Paths below match your local setup.

### 1) Run the benchmark

**PowerShell (Windows):**
```powershell
py -3.13 -m benchmarks.bench_ecg_from_csv `
  --csv "data\MIT-BIH Arrhythmia Database.csv" `
  --limit 50000 `
  --activation-threshold 0.70 `
  --target-rate 0.12 `
  --gate-temperature 0.07 `
  --energy-pressure 0.04 `
  --max-threshold 0.92 `
  --refractory 0 `
  --save results\ecg_bench_50000.json
```

**Typical output (what you observed):**
```
activations               : 5159
activation_rate           : 0.103
energy_remaining          : 89.649
estimated_energy_savings_pct: 85.45% ~ 85.96%
```

### 2) Plot the "energy cost" bar chart
```bash
py -3.13 tools\plot_ecg_bench.py --json results\ecg_bench_50000.json
# writes results\ecg_bench_50000.png
```

### 3) Gallery scripts (optional)

`tools\summarize_and_plot.py` — builds `results\summary.csv`, `summary.md`, and a `results\plots\` set:

- `precision_recall.png`
- `f1_and_rate.png`
- `f1_vs_savings.png`
- `pareto_frontier.png`

## API cheatsheet

### Core types

- **SundewConfig** — dataclass of all knobs (validated via `validate()`).
- **SundewAlgorithm** — the controller/gate.
- **ProcessingResult** — returned when an input is processed (contains `significance`, `processing_time`, `energy_consumed`).

### SundewConfig (key fields)

**Activation & rate control**
- `activation_threshold: float` — starting threshold.
- `target_activation_rate: float` — long-term target fraction to process.
- `ema_alpha: float` — smoothing for the observed activation rate.

**PI controller**
- `adapt_kp, adapt_ki: float` — controller gains.
- `error_deadband: float, integral_clamp: float`.

**Threshold bounds**
- `min_threshold, max_threshold: float`.

**Energy model & gating**
- `energy_pressure: float` — how quickly we tighten when energy drops.
- `gate_temperature: float` — 0 = hard gate; >0 = soft/probing.
- `max_energy, dormant_tick_cost, dormancy_regen, eval_cost, base_processing_cost`.

**Significance weights (sum to 1.0)**
- `w_magnitude, w_anomaly, w_context, w_urgency`.

**Extras**
- `rng_seed: int, refractory: int, probe_every: int`.

You can also load curated presets; see below.

### SundewAlgorithm (most used)
```python
algo = SundewAlgorithm(cfg)
r = algo.process(x: dict[str, float]) -> ProcessingResult | None
rep = algo.report() -> dict[str, float | int]
algo.threshold: float             # live threshold
algo.energy.value: float          # remaining "energy"
```

**Input `x` should contain:**
`magnitude` (0–100 scale), `anomaly_score` [0,1], `context_relevance` [0,1], `urgency` [0,1].

## Configuration presets

Shipped in `sundew.config_presets` and available through helpers:

```python
from sundew import get_preset, list_presets
print(list_presets())
cfg = get_preset("tuned_v2")                  # recommended general-purpose
cfg = get_preset("ecg_v1")                    # ECG-leaning recall
cfg = get_preset("conservative")              # maximize savings
cfg = get_preset("aggressive")                # maximize activations
cfg = get_preset("tuned_v2", {"target_activation_rate": 0.30})
```

The default tuning in `SundewConfig` (as of v0.1.28) is the balanced, modern set you demonstrated:

```python
SundewConfig(
  activation_threshold=0.78, target_activation_rate=0.15,
  gate_temperature=0.08, max_threshold=0.92, energy_pressure=0.04, ...
)
```

## Results you can paste in blogs/papers

**Demo run (50 events):** activation≈0.16, savings≈80.0%, final thr≈0.581, EMA rate≈0.302.

**ECG 50k samples (your run):** activation≈0.103, savings≈85.5%, energy_left≈89.6.

Include your figures:

```markdown
![Precision vs Recall](results/plots/precision_recall.png)
![Activation rate vs F1](results/plots/f1_and_rate.png)
![Savings vs F1](results/plots/f1_vs_savings.png)
![Pareto frontier (F1 vs Savings)](results/plots/pareto_frontier.png)
```

…and the benchmark cost bars:

```markdown
![ECG energy cost](results/ecg_bench_50000.png)
```

## Project structure

```
sundew_algorithms/
├─ src/sundew/                 # library (packaged to PyPI)
│   ├─ cli.py, core.py, energy.py, gating.py, ecg.py
│   ├─ config.py, config_presets.py
│   └─ __main__.py (CLI entry: `python -m sundew`)
├─ benchmarks/                 # repo-only scripts (not shipped to PyPI)
│   └─ bench_ecg_from_csv.py
├─ tools/                      # plotting & summaries
│   ├─ summarize_demo_json.py
│   ├─ plot_significance_hist.py
│   └─ plot_ecg_bench.py
├─ results/ (gitignored)       # JSON runs, plots, CSV summaries
└─ data/    (gitignored)       # local datasets (e.g., MIT-BIH CSV)
```

## License & disclaimer

**MIT License** (see LICENSE)

Research/benchmarking only. Not a medical device; not for diagnosis.

---

### Notes for maintainers

- PyPI is live at 0.1.28; `pip install -U sundew-algorithms==0.1.28` works.
- CI pre-commit: ruff, ruff-format, mypy (src only).
- Future-proofing (optional): move to a SPDX license string in `pyproject.toml` to satisfy upcoming setuptools deprecations.
