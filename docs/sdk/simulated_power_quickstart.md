
# Simulated Power Benchmark Quickstart

This guide walks through running the synthetic workload to estimate power savings
before hardware bring-up.

## Prerequisites
- Python environment with project dependencies installed
- Optional: `matplotlib` and `Pillow` to plot exported results

## Run the simulator
```bash
python benchmarks/power/run_simulated_workload.py --samples 5000 --profile balanced --export results/balanced.json
```

Profiles:
- `aggressive` — higher activation rate (≈40% savings)
- `balanced` — moderate activation (≈60% savings)
- `conservative` — strong gating (≈70% savings)

The command prints headline metrics and writes a JSON export when `--export` is provided.

## Visualize (optional)
```bash
python benchmarks/power/plot_export.py results/balanced.json
```

If `matplotlib`/`Pillow` are missing, the script will explain how to install them.

## What to look for
- `activation_rate` approaching the target value configured in `SDKConfig`
- `estimated_savings` within the 60–80% band for Phase 1 goals
- `events` array for downstream tooling (e.g., telemetry replay, dashboards)
