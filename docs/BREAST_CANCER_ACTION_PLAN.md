# Breast Cancer Feature Enrichment Plan

Current tuning (192 configs in `data/results/breast_cancer_tuning.csv`) fails to keep energy savings >=75% once recall exceeds ~12%. To unlock better trade-offs before hardware validation:

## Immediate prototyping ideas

1. **Augmented anomaly score**
   - Derive aggregate features across mean radius/texture/perimeter etc. for activated events.
   - Add a scaled z-score anomaly feature and feed into Sundew via `anomaly_score_ext`.
   - Update `benchmarks/run_dataset_suite.py` to consume the new column when present.

2. **Probe sampling**
   - Set `probe_every` to ~50 and log outcomes; submitted events can inform adaptive threshold adjustments.
   - Compare energy cost vs recall gains using `sweep_breast_custom_health.py`.

3. **Hybrid post-gate classifier**
   - Reuse `benchmarks/layer_classifier.py` specifically for breast cancer, but apply calibrated thresholds to raise recall without discarding savings.

## Next steps\n\n- Instrument runtime to sample probe_hint activations: log forced gates, measure incremental energy cost, and consider adaptive probe_every (e.g., shrink interval when probe_hint streaks occur).\n
- Generate enriched dataset with new anomaly columns under `data/raw/breast_cancer_*_enriched.csv`.
- Re-run the tuning script with probed configs and log results into `data/results/breast_cancer_tuning_enriched.csv`.
- If recall >=15% with savings >=78% is achieved, schedule hardware validation; otherwise iterate on feature engineering.


- Probe telemetry: enchmarks/run_pipeline_dataset.py on custom_breast_probe (569 samples) yielded 157 activations (27.6%), 72.4% savings, and 19 probe-triggered gates; consider nudging probe_every or anomaly weights before on-device validation.
