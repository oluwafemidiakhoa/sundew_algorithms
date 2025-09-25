# Hardware Validation Plan

## Objectives
- Verify simulated energy savings (`dataset_suite_extended.csv`) against real power measurements.
- Quantify effects of probe-triggered activations before running on-device.

## Preparation
1. Generate enrichment datasets (`data/raw/breast_cancer_wisconsin_enriched.csv`).
2. Capture Sundew telemetry with `benchmarks/run_pipeline_dataset.py` (collect `probe_activations`, activation rate, savings).
3. Build adversarial sequences via `benchmarks/run_adversarial_stream.py` for stress replay.

## Measurement Procedure
1. Stream events to the device under test while logging:
   - Event index, probe flag, activation decision
   - Instantaneous watts / joules (from power meter or emulator)
2. Use `tools/power_capture_template.py` (fill in device-specific API calls) to align Sundew events and power samples.
3. After each run, merge telemetry with power readings and compute measured savings vs simulated baselines.

## Reporting
- Store raw traces in `data/results/hardware/` (CSV + JSON).
- Summarize measured savings with bootstrap intervals using `benchmarks/bootstrap_metrics.py`.
- Note any probe or threshold adjustments made pre/post hardware validation.

- Simulated replay stored in data/results/hardware/merged_probe_power.json (avg watts + probe counts).
