# Runtime Monitoring Hooks

`PipelineRuntime` exposes `add_listener(callback)` to receive per-event telemetry. The callback signature is `(ProcessingResult, component_metrics: Dict[str, Any])`.

Example usage:
```python
from sundew import build_simple_runtime
from sundew.config_presets import get_preset

runtime = build_simple_runtime(get_preset('custom_breast_probe'))

def log_event(result, components):
    if components['energy']['cost'] > 5.0:
        print('High cost event', components['energy'])

runtime.add_listener(log_event)
```

Pair with `benchmarks/run_pipeline_dataset.py` or application pipelines to stream metrics to your observability stack (e.g., logging, Prometheus, etc.).

Recommended alerts:
- `probe_activations` spike above expected median.
- Activation rate deviates >10% from target.
- Energy savings drops below 70%.
