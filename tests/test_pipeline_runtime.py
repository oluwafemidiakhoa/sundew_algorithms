import math

from pathlib import Path

import pandas as pd

from sundew.config import SundewConfig
from sundew.interfaces import ProcessingResult
from sundew.runtime import build_simple_runtime
from sundew.simple_core import SimpleSundewAlgorithm
from sundew import SundewAlgorithm
from sundew.config_presets import get_preset

DATA_DIR = Path("data/raw")


def _make_samples(total: int = 200):
    samples = []
    for i in range(total):
        if i % 5 == 0:
            samples.append(
                {
                    "magnitude": 80,
                    "anomaly_score": 0.8,
                    "urgency": 0.7,
                    "context": 0.6,
                }
            )
        else:
            samples.append(
                {
                    "magnitude": 20,
                    "anomaly_score": 0.2,
                    "urgency": 0.1,
                    "context": 0.2,
                }
            )
    return samples


def test_simple_runtime_tracks_simple_algorithm():
    cfg_pipeline = SundewConfig()
    cfg_simple = SundewConfig()

    runtime = build_simple_runtime(cfg_pipeline)
    baseline = SimpleSundewAlgorithm(cfg_simple)

    samples = _make_samples(250)
    for sample in samples:
        runtime_result = runtime.process(sample)
        baseline_result = baseline.process(sample)

        assert isinstance(runtime_result, ProcessingResult)
        assert runtime_result.activated == (baseline_result is not None)
        assert math.isclose(
            runtime.threshold,
            baseline.threshold,
            rel_tol=1e-6,
            abs_tol=1e-6,
        )

    report_runtime = runtime.report()
    report_simple = baseline.report()

    assert report_runtime["samples_processed"] == report_simple["samples_processed"]
    assert report_runtime["samples_activated"] == report_simple["samples_activated"]
    assert math.isclose(
        report_runtime["activation_rate"],
        report_simple["activation_rate"],
        rel_tol=1e-6,
        abs_tol=1e-6,
    )
    assert math.isclose(
        report_runtime["energy_savings_pct"],
        report_simple["energy_savings_pct"],
        rel_tol=1e-6,
        abs_tol=1e-6,
    )


def test_pipeline_runtime_process_returns_processing_result():
    cfg = SundewConfig()
    runtime = build_simple_runtime(cfg)
    result = runtime.process(
        {
            "magnitude": 90,
            "anomaly_score": 0.9,
            "urgency": 0.8,
            "context": 0.7,
        }
    )

    assert isinstance(result, ProcessingResult)
    assert hasattr(result, "activated")
    assert hasattr(result, "energy_consumed")


def test_custom_health_hd82_meets_recall_and_savings_targets():
    csv_path = DATA_DIR / "uci_heart_disease.csv"
    df = pd.read_csv(csv_path)

    algo = SundewAlgorithm(get_preset("custom_health_hd82"))

    y_true = df["ground_truth"].astype(int).tolist()
    y_pred = []

    feature_keys = ("magnitude", "anomaly_score", "context_relevance", "urgency")
    for _, row in df.iterrows():
        event = {key: float(row[key]) for key in feature_keys}
        result = algo.process(event)
        y_pred.append(1 if result is not None else 0)

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    recall = tp / max(1, tp + fn)

    report = algo.report()
    savings = report.get("estimated_energy_savings_pct", 0.0)
    activation_rate = report.get("activation_rate", 0.0)

    assert 0.18 <= recall <= 0.25, recall
    assert 80.0 <= savings <= 84.0, savings
    assert activation_rate < 0.2, activation_rate
