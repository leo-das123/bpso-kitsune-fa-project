from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier


class FixedNeuralNetwork:
    """
    Fixed one-hidden-layer neural network used for the controlled comparison.

    The same configuration is used for:

        all features -> baseline NN
        BPSO features -> selected-feature NN
    """

    def __init__(
        self,
        hidden_neurons: int = 32,
        learning_rate: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 64,
        random_state: int = 42,
    ) -> None:
        if hidden_neurons < 1:
            raise ValueError(
                "hidden_neurons must be >= 1"
            )
        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be > 0"
            )
        if epochs < 1:
            raise ValueError(
                "epochs must be >= 1"
            )
        if batch_size < 1:
            raise ValueError(
                "batch_size must be >= 1"
            )

        self.hidden_neurons = int(
            hidden_neurons
        )
        self.learning_rate = float(
            learning_rate
        )
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.random_state = int(random_state)

        self.model: MLPClassifier | None = None
        self.training_seconds: float = 0.0
        self.n_features_in_: int | None = None
        self.n_classes_: int | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "FixedNeuralNetwork":
        X = np.asarray(
            X,
            dtype=float,
        )
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2-D feature matrix."
            )

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same number of rows."
            )

        if len(np.unique(y)) < 2:
            raise ValueError(
                "Neural network training requires at least two classes."
            )

        start = time.perf_counter()

        self.model = MLPClassifier(
            hidden_layer_sizes=(
                self.hidden_neurons,
            ),
            activation="relu",
            solver="adam",
            learning_rate_init=self.learning_rate,
            max_iter=self.epochs,
            batch_size=min(
                self.batch_size,
                max(1, len(X)),
            ),
            random_state=self.random_state,
            early_stopping=False,
            shuffle=True,
        )

        self.model.fit(
            X,
            y,
        )

        self.training_seconds = (
            time.perf_counter()
            - start
        )

        self.n_features_in_ = int(
            X.shape[1]
        )
        self.n_classes_ = int(
            len(self.model.classes_)
        )

        return self

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        self._check_ready()
        return np.asarray(
            self.model.predict(X)
        )

    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        self._check_ready()
        return np.asarray(
            self.model.predict_proba(X)
        )

    def predict_with_confidence(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        probabilities = self.predict_proba(X)
        predictions = self.predict(X)
        confidence = probabilities.max(
            axis=1
        )
        return (
            predictions,
            confidence,
        )

    @property
    def loss_curve_(self) -> list[float]:
        self._check_ready()
        return [
            float(value)
            for value in getattr(
                self.model,
                "loss_curve_",
                [],
            )
        ]

    @property
    def classes_(
        self,
    ) -> list[Any]:
        self._check_ready()
        return [
            value
            for value in self.model.classes_
        ]

    def summary(self) -> dict[str, Any]:
        """Return a UI/report-friendly model description."""
        return {
            "model": "MLPClassifier",
            "hidden_layers": 1,
            "hidden_neurons": self.hidden_neurons,
            "activation": "ReLU",
            "optimizer": "Adam",
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
            "features": self.n_features_in_,
            "classes": self.n_classes_,
            "training_seconds": self.training_seconds,
        }

    def get_training_history(self) -> dict[str, Any]:
        return {
            "loss_curve": self.loss_curve_,
            "training_seconds": float(
                self.training_seconds
            ),
        }

    def _check_ready(self) -> None:
        if self.model is None:
            raise RuntimeError(
                "Model is not fitted."
            )

    def save(
        self,
        path: str | Path,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        joblib.dump(
            self,
            path,
        )

    @staticmethod
    def load(
        path: str | Path,
    ) -> "FixedNeuralNetwork":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Neural-network artifact not found: {path}"
            )
        return joblib.load(path)


def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    **kwargs: Any,
) -> FixedNeuralNetwork:
    """Convenience function for the all-feature baseline model."""
    return FixedNeuralNetwork(
        **kwargs
    ).fit(
        X_train,
        y_train,
    )


def train_selected(
    X_train: np.ndarray,
    y_train: np.ndarray,
    selected_indices: list[int] | np.ndarray,
    **kwargs: Any,
) -> FixedNeuralNetwork:
    """Convenience function for the BPSO-selected model."""
    indices = np.asarray(
        selected_indices,
        dtype=int,
    )

    if indices.size == 0:
        raise ValueError(
            "At least one selected feature is required."
        )

    return FixedNeuralNetwork(
        **kwargs
    ).fit(
        X_train[:, indices],
        y_train,
    )
