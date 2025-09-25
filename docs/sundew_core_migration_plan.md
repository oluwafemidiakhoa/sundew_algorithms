# SundewAlgorithm Runtime Migration Plan

## Current Analysis Highlights
- **Significance**: `_compute_significance` mixes weight lookup + noise; currently distinct from runtime linear model. Needs adapter or shared implementation to avoid divergence.
- **Gating**: inline logic combines force-probe, refractory cooldown, hysteresis gating with `gate_probability_with_hysteresis`. Potential stage split into probe manager + probability calculator to keep logic testable.
- **Energy**: `EnergyAccount` drives spend/tick with regen noise; runtime stage must support cap tracking, AIMD nudges, energy pressure signals, and telemetry.
- **Control**: Combination of PI, AIMD, and cap nudges inside `_adapt_threshold`. Will require dedicated stage capturing integral state, EMA activation tracking, and energy pressure interactions.
- **Telemetry**: `Metrics` accumulates multiple histories (threshold, energy, decisions). Runtime needs telemetry bus to keep CLI/tests working.

## Proposed Strategy
1. **Stage Extraction**
   - Build adapters (`LegacyControlPolicy`, `LegacyGatingStrategy`, `LegacyEnergyModel`) that call back into existing private methods so we can validate parity before deeper rewrites.
   - Once parity verified, progressively replace private methods with stage-native implementations.
2. **Runtime Wrapper**
   - Introduce `SundewRuntimeAdapter` that instantiates `PipelineRuntime` with legacy stages and exposes `.process()`/`.report()` aligning with the current API.
   - Swap `SundewAlgorithm` internals to delegate to the runtime while keeping existing attributes for backwards compatibility.
3. **Telemetry Bridge**
   - Mirror current `Metrics` updates by listening to runtime metrics after each `process` call; ensure fields that tests reference stay populated.
4. **Configuration Parity**
   - Ensure builder respects options like `probe_every`, `refractory`, `use_aimd_controller`, energy pressure, etc.
   - Validate defaults and presets to maintain identical behaviour.

## Risk & Regression Checklist
- **Determinism**: random seed usage within legacy algorithm vs runtime adapters; confirm gating probabilities and energy regen remain deterministic for tests.
- **Performance**: abstraction overhead; measure to ensure no drastic slowdown for common workloads.
- **Metrics Drift**: verify all fields in `Metrics` remain accurate, including advanced telemetry histories.
- **CLI/Demo Output**: run `debug_algorithm.py` / CLI demos post-migration to confirm user-facing formatting still works.
- **Energy Accounting**: double-check cumulative energy spent/recovered values align with legacy calculations.
- **Edge Cases**: refractory logic, forced probe cadence, energy cap nudges, and AIMD toggles must still fire as expected.

## Next Steps
- Flesh out TODOs in adapters with real logic and unit tests.
- Create integration regression test feeding identical event streams through legacy vs runtime-backed SundewAlgorithm to assert parity.
- Update CLI/tests to exercise runtime-backed core once adapters validated.
- Implemented LegacyControlPolicy and LegacyGatingStrategy adapters; energy adapter now defers to legacy helpers while avoiding double accounting.
- Dataset suite harness (enchmarks/run_dataset_suite.py) benchmarks five public datasets with energy + classification metrics to ground decisions.
