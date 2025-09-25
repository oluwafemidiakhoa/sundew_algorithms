# Slide Integration Plan: Layered Precision + Energy

Data sources for IoT/MIT-BIH layered classifier visuals:

- `data/results/layered_precision_iot_mitbih.csv` (IoT auto/aggressive, MIT-BIH auto/ecg_best)
- `data/results/dataset_suite_extended.csv` (energy + recall baselines)
- `docs/LAYERED_CLASSIFIER_RESULTS.md` (table ready for copy/paste)

Slide structure:
1. Baseline energy savings table (from dataset report) with callouts for `custom_health_hd82` and `custom_breast_probe`.
2. Overlay chart: precision/recall bars before vs after classifier for IoT + MIT-BIH; quote thresholds from CSV.
3. Footer: mention training on gated activations only, thresholds tuned to keep recall = baseline.

Next actions:
- Generate plot (matplotlib/notebook) using the CSV to visualize precision lift.
- Add narrative on deployment path (classifier sits downstream, no extra energy cost).
- Sync with hardware validation timeline once breast-cancer enrichment run hits =78% savings.

## Plotting Plan

Goal: Display precision uplift (baseline vs layered) with recall held constant.
- Source: `data/results/layered_precision.csv`, `_extended.csv`, `_iot_mitbih.csv`
- Script: `benchmarks/plot_layered_precision.py` (to be created)
- Figure: bar chart per dataset grouping baseline vs layered precision; annotate energy savings from dataset suite CSV.

- Include ssets/layered_precision.png for precision uplift visual.\n- Cite bootstrap intervals from data/results/bootstrap_summary.json (breast probe precision CI 0.301-0.475; heart_hd82 CI 0.679-0.828).\n
