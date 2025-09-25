# Heart Disease custom preset

Parameters tuned around ~82% savings / ~20% recall:

- activation_threshold: 0.56
- target_activation_rate: 0.15
- energy_pressure: 0.02
- gate_temperature: 0.14
- max_threshold: 0.88
- activation_rate (observed): 0.139
- recall (observed): 0.196
- precision (observed): 0.770
- f1 (observed): 0.317
- estimated_energy_savings_pct: 82.04

Derived from `benchmarks/sweep_custom_health_hd82.py` and `data/results/heart_disease_custom_preset.csv`.
