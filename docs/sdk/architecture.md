# Sundew Core SDK Architecture (Draft)

_TODO_: Capture module boundaries, data flow, and integration points with legacy Sundew components.


## IPC Layer
- `ipc.shim` implements the C-facing surface and ScoreEvent helpers.
- `ipc.adapter` bridges protobuf messages with `AdaptiveGateController` + `MetricsTracker`.
- Example usage: `examples/ipc_demo.py` demonstrates an in-process loop before wiring transport.
