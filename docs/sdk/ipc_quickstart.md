
# Sundew IPC Quickstart

1. **Install dependencies** on the target board:
   ```bash
   git clone <repo>
   cd sundew_algorithms
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt grpcio grpcio-tools adafruit-circuitpython-ina219
   ```
2. **Generate bindings** (if not already present):
   ```bash
   python tools/generate_ipc_bindings.py
   ```
3. **Install daemon** using `tools/deploy/ipc_service.sh install` after filling in copy paths.
4. **Validate transport**:
   ```bash
   python tools/send_score_event.py --host 127.0.0.1 --port 8765 --feature glucose_mgdl=140
   ```
   or stream via gRPC using the snippet in `docs/sdk/ipc/README.md`.
5. **Capture power** using `benchmarks/power/capture_power.py` and compare runs via `benchmarks/power/compare_runs.py`.
6. **Commit artifacts**: store CSV logs under `results/power/` and update docs with findings.
