from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .neural_network import FixedNeuralNetwork


@dataclass
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    inference_seconds: float
    confusion_matrix: list[list[int]]
    training_seconds: float = 0.0
    loss_curve: list[float] | None = None
    feature_count: int = 0
    sample_count: int = 0
    labels: list[Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ComparisonResult:
    """Baseline-vs-BPSO comparison used by the results dashboard."""

    baseline: EvaluationResult
    bpso: EvaluationResult
    original_features: int
    selected_features: int
    reduction_percent: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.to_dict(),
            "bpso": self.bpso.to_dict(),
            "original_features": int(self.original_features),
            "selected_features": int(self.selected_features),
            "removed_features": int(
                self.original_features
                - self.selected_features
            ),
            "reduction_percent": float(
                self.reduction_percent
            ),
        }


def _average_mode(
    y_true: np.ndarray,
) -> str:
    """Use binary metrics for binary targets and weighted metrics otherwise."""
    return (
        "binary"
        if len(np.unique(y_true)) == 2
        else "weighted"
    )


def evaluate_model(
    model: FixedNeuralNetwork,
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_count: int | None = None,
    labels: list[Any] | None = None,
) -> EvaluationResult:
    X = np.asarray(
        X,
        dtype=float,
    )
    y = np.asarray(y)

    start = time.perf_counter()
    predictions = model.predict(X)
    elapsed = time.perf_counter() - start

    average = _average_mode(y)

    if average == "binary":
        positive_label = np.unique(y)[-1]
        precision = precision_score(
            y,
            predictions,
            average="binary",
            pos_label=positive_label,
            zero_division=0,
        )
        recall = recall_score(
            y,
            predictions,
            average="binary",
            pos_label=positive_label,
            zero_division=0,
        )
        f1 = f1_score(
            y,
            predictions,
            average="binary",
            pos_label=positive_label,
            zero_division=0,
        )
    else:
        precision = precision_score(
            y,
            predictions,
            average="weighted",
            zero_division=0,
        )
        recall = recall_score(
            y,
            predictions,
            average="weighted",
            zero_division=0,
        )
        f1 = f1_score(
            y,
            predictions,
            average="weighted",
            zero_division=0,
        )

    unique_labels = np.unique(
        np.concatenate(
            [
                np.asarray(y),
                np.asarray(predictions),
            ]
        )
    )

    return EvaluationResult(
        accuracy=float(
            accuracy_score(
                y,
                predictions,
            )
        ),
        precision=float(
            precision
        ),
        recall=float(
            recall
        ),
        f1=float(
            f1
        ),
        inference_seconds=float(
            elapsed
        ),
        confusion_matrix=confusion_matrix(
            y,
            predictions,
            labels=unique_labels,
        ).tolist(),
        training_seconds=float(
            model.training_seconds
        ),
        loss_curve=model.loss_curve_,
        feature_count=int(
            feature_count
            if feature_count is not None
            else X.shape[1]
        ),
        sample_count=int(
            len(X)
        ),
        labels=(
            list(unique_labels)
            if labels is None
            else labels
        ),
    )


def compare_models(
    baseline: FixedNeuralNetwork,
    bpso_model: FixedNeuralNetwork,
    X_test_all: np.ndarray,
    X_test_selected: np.ndarray,
    y_test: np.ndarray,
    *,
    original_features: int,
    selected_features: int,
    class_names: list[Any] | None = None,
) -> ComparisonResult:
    """Evaluate the baseline and selected-feature models on the same test rows."""
    baseline_result = evaluate_model(
        baseline,
        X_test_all,
        y_test,
        feature_count=original_features,
        labels=class_names,
    )

    bpso_result = evaluate_model(
        bpso_model,
        X_test_selected,
        y_test,
        feature_count=selected_features,
        labels=class_names,
    )

    reduction = (
        (original_features - selected_features)
        / original_features
        * 100.0
        if original_features > 0
        else 0.0
    )

    return ComparisonResult(
        baseline=baseline_result,
        bpso=bpso_result,
        original_features=int(
            original_features
        ),
        selected_features=int(
            selected_features
        ),
        reduction_percent=float(
            reduction
        ),
    )


def feature_reduction(
    original_features: int,
    selected_features: int,
) -> dict[str, float | int]:
    removed = max(
        0,
        int(original_features)
        - int(selected_features),
    )

    reduction = (
        removed
        / original_features
        * 100.0
        if original_features > 0
        else 0.0
    )

    return {
        "original_features": int(
            original_features
        ),
        "selected_features": int(
            selected_features
        ),
        "removed_features": int(
            removed
        ),
        "reduction_percent": float(
            reduction
        ),
    }
