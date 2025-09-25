# Sundew Algorithms

Adaptive gating for energy-aware stream processing with reproducible evidence, layered precision uplift, and hardware readiness.

## Highlights
- Probe-aware presets (`custom_health_hd82`, `custom_breast_probe`) balance recall and energy; bootstrap confidence intervals capture statistical certainty.
- Layered classifier optional stage boosts precision to ~1.0 while preserving recall—plot available in `assets/layered_precision.png`.
- Stress-tested via ablations and adversarial streams with results logged in `data/results/ablation_study.csv` and `data/results/adversarial_runs/`.
- Hardware loop prepared: power capture template, runtime probe telemetry, and merge scripts connect simulation with device measurements.

![Layered Precision Uplift](assets/layered_precision.png)

## Run Everything
```
uv run python tools/run_full_evidence.py
```
This triggers the dataset suite, ablations, bootstrap metrics, layered classifier runs, and generates the precision plot.

Key artefacts land in `data/results/` and the main report (`docs/DATASET_BENCHMARK_REPORT.md`).

## Frequent Commands
- Dataset suite: `uv run python benchmarks/run_dataset_suite.py --presets tuned_v2 auto_tuned aggressive conservative energy_saver`
- Probe replay: `uv run python benchmarks/run_pipeline_dataset.py data/raw/breast_cancer_wisconsin_enriched.csv --preset custom_breast_probe --log data/results/runtime_probe_log.json`
- Layered classifier plot: `uv run python benchmarks/plot_layered_precision.py --out assets/layered_precision.png`
- Power capture (simulated): `uv run python tools/power_capture_template.py --events 569 --simulate`
- Merge runtime + power: `uv run python tools/merge_runtime_power.py`

## Evidence Sources
- `docs/DATASET_BENCHMARK_REPORT.md` – main metrics, probe trade-offs, bootstrap CIs.
- `docs/BREAST_CANCER_ACTION_PLAN.md` – enrichment tasks and probe telemetry.
- `docs/HARDWARE_VALIDATION_PLAN.md` & `docs/HARDWARE_REPLAY_CHECKLIST.md` – how to run on-device tests.
- `docs/LAYERED_CLASSIFIER_RESULTS.md` – precision uplift table for slides.
- `docs/STRESS_TEST_REPORT.md` – ablation and adversarial summaries.
- `docs/RUNTIME_MONITORING.md` – listener API and alert guidance.

## Preset Cheat Sheet
| Preset | Dataset | Recall | Savings | Notes |
| --- | --- | --- | --- | --- |
| custom_health_hd82 | Heart disease | ~0.196 | ~82% | Probe-free, bootstrap precision CI 0.679–0.828. |
| custom_breast_probe | Breast cancer | ~0.118 | ~77% | 19 probe activations (logged), enriched features. |
| auto_tuned | IoT/MIT BIH | dataset-dependent | 88–93% | General streaming baseline. |

## Validation Toolbox
- `benchmarks/run_ablation_study.py`
- `benchmarks/run_adversarial_stream.py`
- `benchmarks/bootstrap_metrics.py`

Reports and inspectable outputs live in `data/results/`.

## Hardware Workflow
1. Log Sundew telemetry with `benchmarks/run_pipeline_dataset.py`.
2. Capture watts using `tools/power_capture_template.py` (implement `read_power_sample`).
3. Merge via `tools/merge_runtime_power.py`.
4. Update docs with measured savings.

## Monitoring Hooks
`PipelineRuntime.add_listener(callback)` allows per-event logging. See `tools/runtime_monitor.py` for an example and `docs/RUNTIME_MONITORING.md` for alert ideas.

## Development Notes
- Tests: `uv run pytest`
- Extra deps: `uv pip install hypothesis matplotlib`
- Layered precision CSVs: `data/results/layered_precision*.csv`

## Citation
```
Idiakhoa, O. (2025). Adaptive Threshold Control for Energy-Efficient Stream Processing. Sundew Algorithms.
```
MIT License.
