# Power Telemetry Capture

- Use INA219/INA3221 sensors; log via python script
- Align sampling with IPC activation timeline
- Store results in CSV for benchmarks/power workflows


Capture script: `python benchmarks/power/capture_power.py --duration 300 --interval 0.5 --output board-baseline.csv` then rerun with gating. Use `benchmarks/power/compare_runs.py` to compute savings. Store CSVs under `results/power/` with board/date naming.
