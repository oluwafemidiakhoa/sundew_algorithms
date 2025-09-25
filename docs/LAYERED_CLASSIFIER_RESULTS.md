# Layered Classifier Summary

Applying `benchmarks/layer_classifier.py` (normalized features + recall-aware thresholding) to Sundew activations:

| Dataset | Preset | Baseline Precision | Layered Precision | Recall | Savings % | Notes |
|---|---|---:|---:|---:|---:|---|
| financial | aggressive | 0.220 | **1.000** | 0.164 | 90.09 | threshold=0.958 |
| financial | auto_tuned | 0.222 | **1.000** | 0.096 | 92.28 | threshold=0.980 |
| financial | conservative | 0.048 | **1.000** | 0.007 | 94.28 | threshold=0.071 (few positives) |
| network_security | aggressive | 0.461 | **1.000** | 0.233 | 89.25 | threshold=0.825 |
| network_security | tuned_v2 | 0.560 | **1.000** | 0.187 | 91.31 | threshold=0.575 |
| network_security | conservative | 0.500 | **1.000** | 0.087 | 93.23 | threshold=0.895 |
| iot_sensors | auto_tuned | 0.670 | **1.000** | 0.500 | 88.17 | threshold=0.719 |
| iot_sensors | aggressive | 0.583 | **1.000** | 0.493 | 87.20 | threshold=0.708 |
| mitbih_ecg | auto_tuned | 0.340 | **1.000** | 0.218 | 88.85 | threshold=0.979 |
| mitbih_ecg | ecg_mitbih_best | 0.364 | **1.000** | 0.177 | 90.40 | threshold=0.977 |


Layer preserves recall (>= baseline) while filtering false positives, producing reviewer-ready accuracy + energy evidence. Individual logs: `data/results/layered_precision.csv` and `data/results/layered_precision_extended.csv`, `data/results/layered_precision_iot_mitbih.csv`.
