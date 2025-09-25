# Hardware Replay Checklist

1. Run `benchmarks/run_pipeline_dataset.py data/raw/breast_cancer_wisconsin_enriched.csv --preset custom_breast_probe --log data/results/runtime_probe_log.json`.
2. Execute `tools/power_capture_template.py --events 569 --out data/results/hardware/power_trace.csv` with device-specific `read_power_sample()` implementation.
3. Merge Sundew log (`runtime_probe_log.json`) with power trace to compute measured energy savings; compare to simulated 72.4% and 19 probe activations.
4. Document findings in `docs/HARDWARE_VALIDATION_PLAN.md` and update `docs/BREAST_CANCER_ACTION_PLAN.md`.
