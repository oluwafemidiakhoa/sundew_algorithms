# Phase 1 (2025–2026) — Sundew Core SDK Delivery Plan

## Vision
Deliver the Sundew Core SDK and supporting firmware reference layer to prove energy intelligence on edge/embedded hardware. Success requires a production-quality SDK surface, deployable artifacts for Jetson Nano, Coral Edge TPU, and Raspberry Pi Compute Module, and benchmark evidence of 60–80% power savings per inference cycle.

## Key Outcomes
- `sundew_core_sdk` package exposing adaptive gating, energy accounting, and telemetry hooks.
- Reference firmware/runtime shim enabling C/C++ integration and IPC to the Python SDK.
- Hardware adapters and deployment scripts for Jetson Nano, Coral Edge TPU, Raspberry Pi CM.
- Automated benchmark harness reporting power, latency, activation rate, and accuracy deltas.
- Launch collateral: developer guide, API reference, sample apps, and CI coverage.

## Workstreams

### 1. SDK Architecture & Packaging
- Extract reusable primitives from `src/sundew/` into a versioned SDK package (`src/sdk/` → `sundew_core_sdk`).
- Define stable public API (config, gate controller, telemetry stream, firmware bridge).
- Provide lightweight runtime that swaps between native Sundew and fallback gate.
- Ship Python wheels and initial C++ headers; set up semantic versioning and release scripts.

### 2. Firmware & Embedded Integration
- Design C ABI for gate interactions (init, score update, decision, telemetry push).
- Implement firmware shim in C++ with bindings to the Python runtime (FFI or gRPC microservice).
- Provide board-specific build configs (CMake + cross-compile toolchains) for Jetson, Coral, RPi CM.
- Document deployment pipeline (container images, provisioning scripts, watchdog integration).

### 3. Hardware Enablement & Benchmarking
- Instrument power measurement (INA219/INA3221 or built-in tools) per board.
- Build repeatable benchmark suite measuring power draw, latency, activation rate.
- Target synthetic + real workloads; track 60–80% savings goal with dashboards.
- Integrate results into `benchmarks/` and ensure nightly runs on hardware farm (or emulated fallback).

### 4. Developer Experience & Documentation
- Draft SDK overview, getting started, hardware setup guides (`docs/sdk/`).
- Publish API reference (auto-generated via `pdoc`/`sphinx`) and tutorial notebooks.
- Create sample apps: anomaly detection, adaptive video stream, diabetes monitoring edge demo.
- Establish contribution guidelines and migration notes from existing Sundew modules.

## Milestones & Timeline (2025–2026)

| Quarter | Milestone |
|---------|-----------|
| Q1 2025 | SDK API prototype, minimal firmware shim, Jetson bring-up, unit-test coverage |
| Q2 2025 | Hardware adapters for Coral & RPi, benchmark harness v1, CI cross-build |
| Q3 2025 | Power-savings validation, telemetry dashboard, beta documentation & samples |
| Q4 2025 | Release Candidate: SDK 0.9, firmware packages, integration tests across boards |
| H1 2026 | GA Launch: Sundew Core SDK 1.0 with full docs, support agreements, dev portal |

## Backlog (Initial Sprint Candidates)
9. Protobuf binding generation script + CI check (ensure grpc tools + shared shim builds).
1. Inventory existing modules for extraction (core gating, energy, config).
2. Draft public API surface and namespace layout; formalize ADR.
3. Set up `src/sdk/` skeleton with packaging metadata and CI lint/test jobs.
4. Create firmware interface specification doc + header stub.
5. Stand up Jetson Nano dev container + cross-compile toolchain definitions.
6. Prototype power logging using INA219 on Raspberry Pi CM; capture baseline metrics.
7. Draft benchmark harness interface (power meter, workload, metrics sink).
8. Author "Phase 1 README" summarizing scope and success criteria for internal teams.

## Risks & Mitigations
- **Hardware availability / variance** → procure duplicate boards, document fallback simulation path.
- **Power measurement noise** → plan calibration routine, average over multiple runs.
- **SDK API churn** → formalize design review (ADR) before implementation.
- **Cross-language boundary bugs** → enforce integration tests with hardware-in-loop CI.

## Next Actions
- Approve backlog and assign owners.
- Create repo scaffolding for SDK package and docs.
- Schedule hardware bring-up sessions and procure measurement gear.


### IPC Track Status
- Proto bindings generated with grpcio-tools 1.75.1; tests load real modules.
- Shim handles ScoreEvent conversion and records telemetry.
- Protobuf runtime upgraded to 6.32.1; CI should enforce matching versions.
- Next: integrate gRPC service adapter and hardware shim once firmware API stabilizes.
Refer to docs/sdk/hardware_checklist.md for detailed bring-up schedule.

Next integration steps:
- Implement gRPC transport service using generated bindings.
- Deploy IPC daemon onto Jetson/Coral/RPi per hardware checklist.
- Capture power telemetry comparisons (baseline vs gated) via transport path.

### Completed to date
- SDK scaffolding + metrics + telemetry exports.
- IPC transports: JSON/TCP/Unix socket, CLI client, gRPC service.
- Binding regeneration tooling and unit tests.

### Remaining to finish Phase 1
- Deploy daemon on Jetson/Coral/RPi; validate power savings.
- Integrate gRPC transport into firmware shim and document API surface.
- Publish developer guide & CI automation around new tooling.
Deployment scripts pending: create per-board setup instructions and automation for pushing daemon.

### Phase 1 Acceptance Checklist
- SDK + transports packaged with documentation.
- IPC daemon deployed on Jetson/Coral/RPi with telemetry logs captured.
- Benchmarks demonstrate 60–80% power savings via IPC layer.
- CI validates bindings, transports, and power capture scripts.
- Developer guide published for partners.
Quickstart reference: docs/sdk/ipc_quickstart.md for on-device execution steps.
