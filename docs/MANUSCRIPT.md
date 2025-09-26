# Sundew Algorithms: Energy-Aware Gating for Adaptive Stream Processing

## Abstract
Sundew is an adaptive gating framework designed to maximize energy savings in streaming workloads while preserving recall. This manuscript summarizes the algorithmic design, preset tuning, probe-based exploration, and the evidence pipeline used to validate the system across public datasets. Measurements show 72–93% estimated energy savings with optional layered classifiers boosting precision to 1.0.

## 1. Introduction
- Motivation: edge/IoT deployment constraints, streaming anomaly detection.
- Challenges: balancing recall with battery life, handling drift, reproducibility expectations from stakeholders (e.g., Nvidia).
- Contributions:
  1. Probe-aware presets for domain-specific trade-offs.
  2. Layered classifier stage to increase precision without extra energy cost.
  3. Evidence pipeline (benchmarks, ablations, bootstrap CIs, power instrumentation).
  4. Tooling for hardware replay and monitoring.

## 2. Algorithm Overview
- Core gating logic (significance scoring, PI threshold control, hysteresis).
- Probe sampling mechanism and effective probe cadence.
- Preset catalogue:
  - `custom_health_hd82` (heart disease): target 15% activation, ~82% savings.
  - `custom_breast_probe` (breast cancer): probe-driven recall lift, ~77% savings.
  - `auto_tuned` baseline for IoT/financial streams.

## 3. Validation Methods
- Datasets: breast_cancer_wisconsin_enriched, uci_heart_disease, IoT sensors, MIT-BIH ECG, financial time series, network security.
- Benchmark suite output (`data/results/dataset_suite_extended.csv`).
- Ablation study (`benchmarks/run_ablation_study.py`):
  - Probes vs no probes.
  - Weight rebalancing.
  - Control pressure adjustments.
- Adversarial stress tests (`benchmarks/run_adversarial_stream.py`): spikes, drift, noise.

## 4. Statistical Confidence
- Bootstrap precision/recall intervals (`data/results/bootstrap_summary.json`).
  - Example: `custom_breast_probe` precision 0.386 (95% CI 0.301–0.475).
  - `custom_health_hd82` precision 0.756 (95% CI 0.679–0.828).

## 5. Layered Classifier Uplift
- Logistic layer operating only on gated activations.
- Results across datasets (`docs/LAYERED_CLASSIFIER_RESULTS.md`).
- Plot (`assets/layered_precision.png`) showing baseline vs layered precision with energy annotations.

## 6. Hardware Readiness
- Runtime instrumentation to track `probe_activations`.
- Power capture template (`tools/power_capture_template.py`) + merge script (`tools/merge_runtime_power.py`).
- Checklist (`docs/HARDWARE_REPLAY_CHECKLIST.md`) for empirical measurement.

## 7. Monitoring & Deployment
- Runtime listeners (`PipelineRuntime.add_listener`).
- Suggested alert rules (`docs/RUNTIME_MONITORING.md`): activation drift, probe spike, energy savings drop.
- Guidance for integrating logging into existing observability stacks.

## 8. Reproducibility
- One-command script (`tools/run_full_evidence.py`) regenerates all artefacts.
- README quickstart and `docs/REPRODUCIBILITY.md` document the workflow.
- Repository layout and CI checks (ruff, mypy, pytest).

## 9. Conclusion
- Sundew achieves high energy savings while maintaining recall using domain-specific presets and probes.
- Layered classifiers deliver precision >0.99 without additional activation cost.
- The comprehensive evidence/stress/hardware tooling prepares Sundew for stakeholder audits and real deployments.

## References
- Idiakhoa, O. (2025). *Adaptive Threshold Control for Energy-Efficient Stream Processing*. Sundew Algorithms project documentation.
