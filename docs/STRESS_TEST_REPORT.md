# Stress & Failure Analysis Summary

## Ablation Highlights

See `data/results/ablation_study.csv` (generated via `benchmarks/run_ablation_study.py`). Key findings:
- Removing probes from `custom_breast_probe` drops recall to 9.0% while savings rise to ~79%, confirming probes drive the accuracy uplift.
- Reweighting anomaly/context (holding probes) keeps recall ~11.2% with ~77% savings.
- Higher energy pressure for `custom_health_hd82` maintains recall (~20%) but does not materially improve savings.
- Raising the IoT threshold slightly increases recall to ~0.513 yet preserves ~88% savings.

## Adversarial Streams

`benchmarks/run_adversarial_stream.py` simulates spikes, drift, and noise. Sample drift run (`custom_health_hd82`, 1,500 events) produced:
- Activation rate: 23.7% (target 15%)
- Energy savings: 76.3%
- Probe activations: 0 (preset has no probes)

Outputs live in `data/results/adversarial_runs/` for repeatable review.

## Next Steps
- Add spikes/noise runs for `custom_breast_probe` once probe instrumentation is finalized.
- Feed these CSVs into statistical tooling (bootstrap analysis) to attach confidence intervals.
