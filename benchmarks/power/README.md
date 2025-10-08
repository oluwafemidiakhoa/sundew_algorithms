# Power Benchmarks (Draft)

Scripts in this folder will collect power, latency, and activation metrics across Jetson Nano, Coral Edge TPU, and Raspberry Pi Compute Module.

Planned components:
- `collect_power.py` — capture raw sensor data (INA219/INA3221).
- `run_workload.py` — orchestrate inference workloads with adjustable gating.
- `report.py` — aggregate and visualize savings relative to baseline.
