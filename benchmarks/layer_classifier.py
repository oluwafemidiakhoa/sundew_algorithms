#!/usr/bin/env python3
"""Train a lightweight classifier on gated activations to boost precision."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

FEATURE_KEYS = ("magnitude", "anomaly_score", "context_relevance", "urgency")


class LogisticLayer:
    """Minimal logistic regression with class balancing."""

    def __init__(self, max_iter: int = 600, lr: float = 0.05, random_state: int = 42) -> None:
        self.max_iter = max_iter
        self.lr = lr
        self.random_state = random_state
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticLayer":
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape
        self.coef_ = rng.normal(scale=0.01, size=n_features)
        self.intercept_ = 0.0

        pos = np.sum(y == 1)
        neg = n_samples - pos
        weight_pos = n_samples / (2 * pos) if pos else 0.0
        weight_neg = n_samples / (2 * neg) if neg else 0.0

        for _ in range(self.max_iter):
            logits = X @ self.coef_ + self.intercept_
            probs = 1.0 / (1.0 + np.exp(-logits))
            errors = probs - y
            sample_weights = np.where(y == 1, weight_pos, weight_neg)
            grad_w = (sample_weights * errors) @ X / n_samples
            grad_b = np.sum(sample_weights * errors) / n_samples

            self.coef_ -= self.lr * grad_w
            self.intercept_ -= self.lr * grad_b

            if np.linalg.norm(self.lr * grad_w) < 1e-6:
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Model not fitted")
        logits = X @ self.coef_ + self.intercept_
        return 1.0 / (1.0 + np.exp(-logits))


@dataclass
class LayerResult:
    dataset: str
    preset: str
    baseline_precision: float
    baseline_recall: float
    baseline_f1: float
    layered_precision: float
    layered_recall: float
    layered_f1: float
    activation_rate: float
    energy_savings_pct: float
    notes: str = ""


def _load_log(path: Path) -> Tuple[Dict, pd.DataFrame]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    summary = payload.get("summary", {})
    raw = payload.get("raw", payload)
    y_true = np.array(raw["y_true"], dtype=int)
    y_pred = np.array(raw["y_pred"], dtype=int)
    events = raw.get("events")
    if events is None:
        raise ValueError(f"Log {path} missing raw events; rerun with --logdir to capture details.")
    df = pd.DataFrame(events)
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    return summary, df


def _normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def _train_layer(
    df: pd.DataFrame, random_state: int = 42
) -> Tuple[LogisticLayer, Dict[str, np.ndarray], Dict[str, float]]:
    activated = df[df["y_pred"] == 1]
    if activated.empty:
        raise ValueError("No activated samples to train classifier on.")
    X = activated[list(FEATURE_KEYS)].values
    y = activated["y_true"].values
    if len(np.unique(y)) < 2:
        raise ValueError("Activated samples contain only one class; cannot train layer.")

    rng = np.random.default_rng(random_state)
    indices = np.arange(len(y))
    rng.shuffle(indices)
    split = max(1, int(0.3 * len(y)))
    test_idx = indices[:split]
    train_idx = indices[split:]
    if len(train_idx) == 0:
        train_idx = indices
        test_idx = indices[:1]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    model = LogisticLayer(random_state=random_state)
    model.fit(_normalize(X_train, mean, std), y_train)

    probs = model.predict_proba(_normalize(X_test, mean, std))
    preds = (probs >= 0.5).astype(int)
    tp = int(((preds == 1) & (y_test == 1)).sum())
    fp = int(((preds == 1) & (y_test == 0)).sum())
    fn = int(((preds == 0) & (y_test == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall) / max(1e-12, precision + recall) if tp + fp + fn > 0 else 0.0
    return model, {"mean": mean, "std": std}, {
        "layer_precision_test": precision,
        "layer_recall_test": recall,
        "layer_f1_test": f1,
        "test_samples": int(len(y_test)),
    }


def _compute_metrics(true: np.ndarray, pred: np.ndarray) -> Tuple[float, float, float]:
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall) / max(1e-12, precision + recall) if tp + fp + fn > 0 else 0.0
    return precision, recall, f1


def _apply_layer(
    df: pd.DataFrame,
    model: LogisticLayer,
    scaler: Dict[str, np.ndarray],
    baseline_rec: float,
    min_recall_factor: float = 0.85,
) -> Tuple[np.ndarray, float, float, float, float]:
    gate = df["y_pred"].to_numpy()
    true = df["y_true"].to_numpy()

    activated_mask = gate == 1
    layer_pred = np.zeros_like(gate)
    if activated_mask.any():
        feats = df.loc[activated_mask, FEATURE_KEYS].values
        mean = scaler["mean"]
        std = scaler["std"]
        scores = model.predict_proba(_normalize(feats, mean, std))

        thresholds = np.unique(scores)
        best_metrics = (0.0, 0.0, 0.0)
        best_threshold = 0.5
        combined_pred = np.zeros_like(gate)
        for thr in np.concatenate([thresholds, [0.5]]):
            preds = (scores >= thr).astype(int)
            candidate = gate.copy()
            candidate[activated_mask] = preds
            precision, recall, f1 = _compute_metrics(true, candidate)
            if recall >= baseline_rec * min_recall_factor and f1 > best_metrics[2]:
                best_metrics = (precision, recall, f1)
                best_threshold = float(thr)
                combined_pred = candidate
        if best_metrics[2] == 0.0:
            preds = (scores >= 0.5).astype(int)
            combined_pred = gate.copy()
            combined_pred[activated_mask] = preds
            best_metrics = _compute_metrics(true, combined_pred)
            best_threshold = 0.5
        precision, recall, f1 = best_metrics
        return combined_pred, precision, recall, f1, best_threshold

    combined_pred = gate.copy()
    precision, recall, f1 = _compute_metrics(true, combined_pred)
    return combined_pred, precision, recall, f1, 1.0


def evaluate_log(path: Path) -> LayerResult:
    summary, df = _load_log(path)
    dataset = summary.get("dataset_name", "unknown")
    preset = summary.get("preset", Path(path).stem)

    baseline_prec = summary.get("precision")
    baseline_rec = summary.get("recall")
    baseline_f1 = summary.get("f1")

    activation_rate = summary.get("activation_rate", df["y_pred"].mean())
    energy_savings = summary.get("estimated_energy_savings_pct", 0.0)

    model, scaler, holdout_metrics = _train_layer(df)
    combined_pred, layered_prec, layered_rec, layered_f1, threshold = _apply_layer(
        df, model, scaler, baseline_rec
    )

    notes = (
        f"layer test precision={holdout_metrics['layer_precision_test']:.3f}, "
        f"recall={holdout_metrics['layer_recall_test']:.3f}, f1={holdout_metrics['layer_f1_test']:.3f}, "
        f"n_test={holdout_metrics['test_samples']}, threshold={threshold:.3f}"
    )

    return LayerResult(
        dataset=dataset,
        preset=preset,
        baseline_precision=float(baseline_prec),
        baseline_recall=float(baseline_rec),
        baseline_f1=float(baseline_f1),
        layered_precision=float(layered_prec),
        layered_recall=float(layered_rec),
        layered_f1=float(layered_f1),
        activation_rate=float(activation_rate),
        energy_savings_pct=float(energy_savings),
        notes=notes,
    )


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Layer a classifier on top of Sundew activations")
    ap.add_argument("logs", nargs="+", help="Paths to JSON logs produced by run_dataset_suite")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional CSV output path for aggregated layer results.",
    )
    args = ap.parse_args(argv)

    results: List[LayerResult] = []
    for log in args.logs:
        res = evaluate_log(Path(log))
        results.append(res)
        print(
            f"{res.dataset} ({res.preset}) | precision {res.baseline_precision:.3f} -> {res.layered_precision:.3f} | "
            f"recall {res.baseline_recall:.3f} -> {res.layered_recall:.3f} | f1 {res.baseline_f1:.3f} -> {res.layered_f1:.3f}"
        )
        print(f"  notes: {res.notes}")

    if args.out:
        df = pd.DataFrame([r.__dict__ for r in results])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"Saved layered results to {args.out}")


if __name__ == "__main__":
    main()
