import math
import random

import pytest

from sundew import SundewAlgorithm, SundewConfig
from sundew.runtime import build_legacy_runtime


def _stream(count: int = 250):
    samples = []
    for i in range(count):
        if i % 7 == 0:
            samples.append(
                {
                    "magnitude": 85,
                    "anomaly_score": 0.75,
                    "context_relevance": 0.6,
                    "urgency": 0.55,
                }
            )
        elif i % 5 == 0:
            samples.append(
                {
                    "magnitude": 40,
                    "anomaly_score": 0.35,
                    "context_relevance": 0.3,
                    "urgency": 0.25,
                }
            )
        else:
            samples.append(
                {
                    "magnitude": 20,
                    "anomaly_score": 0.1,
                    "context_relevance": 0.15,
                    "urgency": 0.1,
                }
            )
    return samples


def _collect_trace(algo: SundewAlgorithm, stream):
    activations = []
    thresholds = []
    energies = []
    energy_spent = []
    energy_recovered = []

    for sample in stream:
        result = algo.process(sample)
        activations.append(result is not None)
        thresholds.append(algo.threshold)
        energies.append(algo.energy.value)
        energy_spent.append(algo.metrics.cumulative_energy_spent)
        energy_recovered.append(algo.metrics.cumulative_energy_recovered)

    return {
        "activations": activations,
        "thresholds": thresholds,
        "energies": energies,
        "energy_spent": energy_spent,
        "energy_recovered": energy_recovered,
        "processed": algo.metrics.processed,
        "activated": algo.metrics.activated,
        "processing_time": algo.metrics.total_processing_time,
    }


def test_legacy_runtime_parity():
    cfg = SundewConfig()
    samples = _stream()

    random.seed(cfg.rng_seed)
    legacy_algo = SundewAlgorithm(cfg)
    legacy_trace = _collect_trace(legacy_algo, samples)

    random.seed(cfg.rng_seed)
    runtime_algo = SundewAlgorithm(cfg)
    runtime = build_legacy_runtime(runtime_algo)

    runtime_trace = {
        "activations": [],
        "thresholds": [],
        "energies": [],
        "energy_spent": [],
        "energy_recovered": [],
    }

    for sample in samples:
        result = runtime.process(sample)
        runtime_trace["activations"].append(result.activated)
        runtime_trace["thresholds"].append(runtime_algo.threshold)
        runtime_trace["energies"].append(runtime_algo.energy.value)
        runtime_trace["energy_spent"].append(runtime_algo.metrics.cumulative_energy_spent)
        runtime_trace["energy_recovered"].append(runtime_algo.metrics.cumulative_energy_recovered)

    assert runtime_algo.metrics.processed == legacy_trace["processed"]
    assert runtime_algo.metrics.activated == legacy_trace["activated"]
    assert math.isclose(
        runtime_algo.metrics.total_processing_time,
        legacy_trace["processing_time"],
        rel_tol=1e-7,
        abs_tol=1e-7,
    )

    for key in ("activations", "thresholds", "energies", "energy_spent", "energy_recovered"):
        runtime_series = runtime_trace[key]
        legacy_series = legacy_trace[key]
        assert len(runtime_series) == len(legacy_series)
        for r_val, l_val in zip(runtime_series, legacy_series):
            if isinstance(r_val, bool):
                assert r_val is l_val
            else:
                assert math.isclose(r_val, l_val, rel_tol=1e-7, abs_tol=1e-7)
