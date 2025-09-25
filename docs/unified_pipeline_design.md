# Unified Sundew Pipeline Architecture

## Objectives
- Provide a single runtime that can host both the current `SundewAlgorithm` behaviour and the pluggable stack from `EnhancedSundewAlgorithm`.
- Make control, gating, energy, and learning modules interchangeable at runtime through configuration instead of code forks.
- Preserve the simplified API surface exposed through `sundew.__init__` while enabling advanced users to compose richer pipelines.
- Create a foundation for adaptive, hardware-in-the-loop, and distributed extensions without duplicating logic.

## Current State & Gaps
- `core.py` implements a tightly-coupled event loop with inlined PI control, AIMD tweaks, and direct calls to the energy account.
- `enhanced_core.py` sketches a modular architecture but is not wired into the package exports or exercised by tests.
- The simplified engine in `simple_core.py` diverges from the main runtime, forcing tests and demos to choose between inconsistent behaviours.
- Telemetry, CLI, and demos depend on Core-specific structures, making it hard to reuse them for the enhanced modules.

## Proposed Unified Architecture
1. **Pipeline Driver**
   - A new `PipelineRuntime` orchestrates the event processing loop.
   - Responsibilities: maintain global metrics, broker stage execution, manage lifecycle hooks (initialise, process, report, shutdown).
2. **Stage Interfaces**
   - Standardise the signatures already implied in `interfaces.py` for
     significance models, gating strategies, control policies, energy models, and learning modules.
   - Add light adapters so legacy `SundewAlgorithm` components can be wrapped without rewriting them immediately.
3. **Composition Graph**
   - Represent pipeline layout via configuration (YAML/JSON) describing which stage implementations to instantiate plus their params.
   - Support defaults mirroring today’s behaviour for backwards compatibility.
4. **Telemetry Bus**
   - Centralise metric/trace emission so CLI, monitoring dashboards, and tests can consume consistent structures regardless of pipeline flavour.
5. **Compatibility Layer**
   - Keep `SundewAlgorithm` as a thin facade around the unified runtime with the “classic” preset.
   - Provide `SimpleSundewAlgorithm` via the same runtime but with minimal stage set.
   - Expose the enhanced plugin catalogue through presets and the CLI.

## Component Responsibilities
- **Ingress Adapter**: normalises raw feature dicts, injects timestamps/ids, and seeds `ProcessingContext` (optional for simple mode).
- **Significance Stage**: delegates to configured model (linear, neural, external) to score the event and annotate the context.
- **Energy Stage**: consults the chosen energy model to determine current budget, regen, costs, and apply spend/regain.
- **Gating Stage**: computes gate probability/decision using configured strategy, respecting hysteresis/AIMD hooks where enabled.
- **Control Stage**: updates thresholds or other policy variables based on metrics and energy signals; pushes adjustments back into runtime state.
- **Learning Stage (optional)**: consumes experiences to update models and notify control stage about new tuning recommendations.
- **Telemetry Stage**: records decisions, controller errors, energy stats, and publishes structured events at configured cadence.

## Configuration & Composition
- Introduce `sundew.config_runtime` module with:
  - `RuntimeConfig` dataclass capturing stage selections, parameters, and compatibility flags.
  - Loader helpers for YAML/JSON plus programmatic builders.
- Map friendly preset names to config bundles (reuse `config_presets.py` machinery).
- Ensure default construction mirrors current `SundewAlgorithm` semantics so existing callers require no change.

## Refactor Plan
1. **Scaffolding**
   - Implement `PipelineRuntime` with the classic stack hard-coded but routed through stage interfaces.
   - Wrap existing core logic into stage classes (e.g., `ClassicControlStage`, `ClassicEnergyStage`).
2. **Modularisation**
   - Refactor `enhanced_core.py` components to conform to the shared interfaces.
   - Replace bespoke loops with the runtime driver; delete duplicate state once verified.
3. **Backwards Compatibility**
   - Update CLI, demos, and tests to instantiate the runtime through a compatibility wrapper.
   - Maintain existing result/metric structures where public APIs depend on them.
4. **Advanced Feature Enablement**
   - Re-enable the advanced modules (information theory, adaptive learning, AutoML) by registering them as optional stages.

## Testing Strategy
- Extend unit tests to cover each stage contract individually.
- Add integration suites that execute the same input streams across "classic" and "enhanced" presets, asserting equivalent outcomes where expected.
- Introduce scenario tests for configuration parsing and error handling (invalid stage names, missing dependencies, etc.).
- Enhance demo/CLI smoke tests to ensure telemetry output is stable post-refactor.

## Risks & Mitigations
- **Regression Risk**: Large refactor could break existing behaviour.
  - Mitigate via golden-path regression tests and incremental stage extraction.
- **Performance Regressions**: Added abstraction layers may increase overhead.
  - Profile hot path; default to lightweight dataclasses and avoid unnecessary object churn.
- **Dependency Sprawl**: Advanced modules pull in heavy packages.
  - Keep optional extras isolated and lazily imported; document extras in `pyproject.toml`.
- **Configuration Complexity**: Rich pipeline configs may overwhelm users.
  - Ship opinionated presets and provide CLI/GUI tooling for editing configs.

## Open Questions
- How should concurrency (batch processing, GPU) integrate with the single-event driver?
- What persistence, if any, is needed for adaptive learning checkpoints?
- Which telemetry sink format best serves both the CLI and remote monitoring (JSON lines, OpenTelemetry, etc.)?
- Do we guarantee determinism when combining stochastic stages (e.g., adaptive learning + energy regen noise)?
\n## Migration Progress\n- SimpleSundewAlgorithm now wraps the shared PipelineRuntime (see src/sundew/simple_core.py) ensuring parity while consolidating logic.\n- A dedicated regression test (tests/test_pipeline_runtime.py) checks behaviour matches the prior implementation.
- Added legacy adapters stub in src/sundew/runtime/core_adapter.py to map SundewAlgorithm components onto the runtime.
- Legacy control and gating adapters scaffolded in src/sundew/runtime/core_adapter.py to preserve existing behaviour during migration.
- SundewAlgorithm now delegates to LegacyRuntimeAdapter, eliminating duplicate hot-path logic.
