# Reproducibility Guide

Run the full evidence suite with one command:

```
uv run python tools/run_full_evidence.py
```

This executes:
1. Dataset suite (`benchmarks/run_dataset_suite.py`) – refreshes CSV/JSON metrics.
2. Ablation study (`benchmarks/run_ablation_study.py`).
3. Bootstrap confidence intervals (`benchmarks/bootstrap_metrics.py`).
4. Layered classifier evaluation (`benchmarks/layer_classifier.py`).
5. Precision uplift plot (`benchmarks/plot_layered_precision.py`).

Outputs are written to `data/results/` and `assets/layered_precision.png`. Review these artefacts before sharing with stakeholders.
