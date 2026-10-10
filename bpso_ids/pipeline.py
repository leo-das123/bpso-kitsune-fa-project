from __future__ import annotations

from pathlib import Path
import json
import time
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

from .bpso import BPSOResult, BinaryPSO
from .config import ProjectConfig
from .evaluation import (
    ComparisonResult,
    compare_models,
    evaluate_model,
    feature_reduction,
)
from .fuzzy import FuzzyConfig, FuzzyRiskEngine
from .neural_network import FixedNeuralNetwork
from .preprocessing import PreparedData, Preprocessor
from .schema import detect_target_candidates, profile_dataset


class ResearchPipeline:
    """
    Master research-engine pipeline.

    The pipeline is intentionally stage-based so NiceGUI can run and display
    each stage independently:

        analyze_dataset()
        prepare_dataset()
        train_baseline()
        initialize_bpso()
        run_bpso()
        train_selected_model()
        configure_fuzzy()
        run_fuzzy()
        evaluate()
        predict_record()
        export_report()
    """

    def __init__(
        self,
        config: ProjectConfig,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.progress_callback = progress_callback

        self.preprocessor = Preprocessor(
            random_state=config.random_state,
            test_size=config.test_size,
            validation_size=config.validation_size,
            max_classes=config.max_classes,
        )

        self.raw_dataframe: pd.DataFrame | None = None
        self.dataset_profile: dict[str, Any] | None = None
        self.prepared: PreparedData | None = None

        self.baseline: FixedNeuralNetwork | None = None
        self.selected_model: FixedNeuralNetwork | None = None

        self.bpso_optimizer: BinaryPSO | None = None
        self.bpso_result: BPSOResult | None = None

        self.selected_indices: list[int] = []
        self.selected_features: list[str] = []

        self.fuzzy_engine: FuzzyRiskEngine | None = None
        self.fuzzy_features: list[str] = []
        self.fuzzy_indices: list[int] = []
        self.fuzzy_min: np.ndarray | None = None
        self.fuzzy_max: np.ndarray | None = None

        self.comparison: ComparisonResult | None = None
        self.baseline_metrics: dict[str, Any] | None = None
        self.selected_metrics: dict[str, Any] | None = None

        self.last_fitness_details: dict[str, Any] = {}
        self.last_bpso_state: dict[str, Any] = {}
        self.stage_status: dict[str, str] = {
            "dataset": "pending",
            "preprocessing": "pending",
            "baseline": "pending",
            "bpso": "pending",
            "selected_model": "pending",
            "fuzzy": "pending",
            "evaluation": "pending",
            "report": "pending",
        }

    # ------------------------------------------------------------------
    # Progress/state
    # ------------------------------------------------------------------

    def _emit(
        self,
        event: str,
        **payload: Any,
    ) -> None:
        state = {
            "event": event,
            "stage_status": dict(
                self.stage_status
            ),
            **payload,
        }

        if self.progress_callback is not None:
            self.progress_callback(state)

    # ------------------------------------------------------------------
    # Stage 1: dataset analysis
    # ------------------------------------------------------------------

    def analyze_dataset(
        self,
        df: pd.DataFrame,
        target_column: str | None = None,
    ) -> dict[str, Any]:
        self.raw_dataframe = df.copy()

        profile = profile_dataset(
            self.raw_dataframe,
            target_column=target_column,
        )

        self.dataset_profile = profile.to_dict()
        self.stage_status["dataset"] = "complete"

        self._emit(
            "dataset_analyzed",
            profile=self.dataset_profile,
        )

        return self.dataset_profile

    def resolve_target(
        self,
        explicit_target: str | None = None,
    ) -> str:
        if explicit_target:
            if self.raw_dataframe is None:
                raise RuntimeError(
                    "Dataset must be loaded before resolving a target."
                )

            if explicit_target not in self.raw_dataframe.columns:
                raise ValueError(
                    f"Target column not found: {explicit_target}"
                )

            return str(explicit_target)

        if self.config.target_column:
            return str(
                self.config.target_column
            )

        if self.dataset_profile:
            candidates = self.dataset_profile.get(
                "possible_target_columns",
                [],
            )
            if candidates:
                return str(
                    candidates[0]
                )

        if self.raw_dataframe is not None:
            candidates = detect_target_candidates(
                self.raw_dataframe
            )
            if candidates:
                return str(
                    candidates[0]
                )

        raise ValueError(
            "No target column could be detected. "
            "Select the target column explicitly."
        )

    # ------------------------------------------------------------------
    # Stage 2: preprocessing
    # ------------------------------------------------------------------

    def prepare_dataset(
        self,
        target_column: str | None = None,
        drop_columns: list[str] | None = None,
    ) -> PreparedData:
        if self.raw_dataframe is None:
            raise RuntimeError(
                "Load a dataframe before preprocessing."
            )

        target = self.resolve_target(
            target_column
        )

        requested_drop = (
            self.config.drop_columns
            if drop_columns is None
            else drop_columns
        )

        self.prepared = self.preprocessor.fit_prepare(
            self.raw_dataframe,
            target_column=target,
            drop_columns=requested_drop,
            max_rows=self.config.max_rows,
        )

        if self.dataset_profile is None:
            self.analyze_dataset(
                self.raw_dataframe,
                target_column=target,
            )

        self.dataset_profile["target_column"] = target
        self.dataset_profile["dropped_columns"] = list(
            self.preprocessor.dropped_columns
        )
        self.dataset_profile["usable_features"] = len(
            self.prepared.feature_names
        )
        self.dataset_profile["train_rows"] = len(
            self.prepared.X_train
        )
        self.dataset_profile["validation_rows"] = len(
            self.prepared.X_val
        )
        self.dataset_profile["test_rows"] = len(
            self.prepared.X_test
        )
        self.dataset_profile["classes"] = list(
            self.prepared.class_names
        )

        self.stage_status["preprocessing"] = "complete"

        self._emit(
            "dataset_prepared",
            prepared=self.prepared,
            profile=self.dataset_profile,
        )

        return self.prepared

    # ------------------------------------------------------------------
    # Compatibility loading helpers
    # ------------------------------------------------------------------

    def load_dataframe(
        self,
        df: pd.DataFrame,
        target_column: str | None = None,
        drop_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        self.analyze_dataset(
            df,
            target_column=target_column,
        )
        self.prepare_dataset(
            target_column=target_column,
            drop_columns=drop_columns,
        )
        return df

    def load_and_prepare(
        self,
        dataset_csv: str | Path,
        target_column: str | None = None,
        drop_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        dataset_csv = Path(
            dataset_csv
        )

        if not dataset_csv.exists():
            raise FileNotFoundError(
                f"Dataset not found: {dataset_csv}"
            )

        df = pd.read_csv(
            dataset_csv
        )

        return self.load_dataframe(
            df,
            target_column=target_column,
            drop_columns=drop_columns,
        )

    # ------------------------------------------------------------------
    # Stage 3: baseline neural network
    # ------------------------------------------------------------------

    def train_baseline(self) -> dict[str, Any]:
        if self.prepared is None:
            raise RuntimeError(
                "Prepare the dataset before training the baseline model."
            )

        self.baseline = FixedNeuralNetwork(
            hidden_neurons=self.config.hidden_neurons,
            learning_rate=self.config.learning_rate,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            random_state=self.config.random_state,
        ).fit(
            self.prepared.X_train,
            self.prepared.y_train,
        )

        validation = evaluate_model(
            self.baseline,
            self.prepared.X_val,
            self.prepared.y_val,
            feature_count=len(
                self.prepared.feature_names
            ),
            labels=self.prepared.class_names,
        )

        test = evaluate_model(
            self.baseline,
            self.prepared.X_test,
            self.prepared.y_test,
            feature_count=len(
                self.prepared.feature_names
            ),
            labels=self.prepared.class_names,
        )

        self.baseline_metrics = {
            "validation": validation.to_dict(),
            "test": test.to_dict(),
            "model_summary": self.baseline.summary(),
        }

        self.stage_status["baseline"] = "complete"

        self._emit(
            "baseline_trained",
            metrics=self.baseline_metrics,
        )

        return self.baseline_metrics

    # ------------------------------------------------------------------
    # Stage 4: BPSO fitness / initialization
    # ------------------------------------------------------------------

    def _fitness_function(
        self,
        position: np.ndarray,
    ) -> tuple[float, dict[str, Any]]:
        if self.prepared is None:
            raise RuntimeError(
                "Prepared data is required for BPSO fitness evaluation."
            )

        indices = np.flatnonzero(
            position
        )

        if len(indices) < min(
            self.config.min_selected_features,
            len(position),
        ):
            return (
                float("-inf"),
                {
                    "validation_f1": float("nan"),
                    "selected_features": int(
                        len(indices)
                    ),
                    "total_features": int(
                        len(position)
                    ),
                    "feature_penalty": float("nan"),
                    "fitness": float("-inf"),
                },
            )

        X_train = self.prepared.X_train[:, indices]
        X_val = self.prepared.X_val[:, indices]

        # Demo mode uses a smaller fitness-training workload so the interactive
        # UI remains practical. Research mode uses the prepared training set.
        use_demo_limits = (
            getattr(self.config, "mode", "research") == "demo"
        )

        epochs = (
            min(
                self.config.epochs,
                self.config.demo_fitness_epochs,
            )
            if use_demo_limits
            else self.config.epochs
        )

        if use_demo_limits:
            max_rows = min(
                len(X_train),
                self.config.demo_fitness_max_rows,
            )
        else:
            max_rows = len(X_train)

        if max_rows < len(X_train):
            seed_offset = int(
                np.flatnonzero(position).sum()
            )
            rng = np.random.default_rng(
                self.config.random_state
                + seed_offset
            )
            selected_rows = rng.choice(
                len(X_train),
                size=max_rows,
                replace=False,
            )
            X_fit = X_train[
                selected_rows
            ]
            y_fit = self.prepared.y_train[
                selected_rows
            ]
        else:
            X_fit = X_train
            y_fit = self.prepared.y_train

        candidate_model = FixedNeuralNetwork(
            hidden_neurons=self.config.hidden_neurons,
            learning_rate=self.config.learning_rate,
            epochs=epochs,
            batch_size=self.config.batch_size,
            random_state=self.config.random_state,
        )

        candidate_model.fit(
            X_fit,
            y_fit,
        )

        validation = evaluate_model(
            candidate_model,
            X_val,
            self.prepared.y_val,
            feature_count=len(indices),
            labels=self.prepared.class_names,
        )

        feature_ratio = (
            len(indices)
            / len(position)
        )

        penalty = (
            self.config.feature_penalty_lambda
            * feature_ratio
        )

        fitness = float(
            validation.f1
            - penalty
        )

        details = {
            "validation_f1": float(
                validation.f1
            ),
            "validation_accuracy": float(
                validation.accuracy
            ),
            "validation_precision": float(
                validation.precision
            ),
            "validation_recall": float(
                validation.recall
            ),
            "selected_features": int(
                len(indices)
            ),
            "total_features": int(
                len(position)
            ),
            "feature_ratio": float(
                feature_ratio
            ),
            "feature_penalty": float(
                penalty
            ),
            "fitness": fitness,
            "fitness_training_rows": int(
                len(X_fit)
            ),
            "fitness_epochs": int(
                epochs
            ),
        }

        self.last_fitness_details = dict(
            details
        )

        return fitness, details

    def initialize_bpso(self) -> BinaryPSO:
        if self.prepared is None:
            raise RuntimeError(
                "Prepare the dataset before initializing BPSO."
            )

        self.bpso_optimizer = BinaryPSO(
            n_features=len(
                self.prepared.feature_names
            ),
            fitness_fn=self._fitness_function,
            n_particles=self.config.n_particles,
            n_iterations=self.config.n_iterations,
            inertia_weight=self.config.inertia_weight,
            cognitive_coeff=self.config.cognitive_coeff,
            social_coeff=self.config.social_coeff,
            min_selected_features=min(
                self.config.min_selected_features,
                len(self.prepared.feature_names),
            ),
            random_state=self.config.random_state,
            progress_callback=(
                self._bpso_callback
                if self.config.fitness_callback_enabled
                else None
            ),
        )

        self._emit(
            "bpso_initialized",
            n_features=len(
                self.prepared.feature_names
            ),
            n_particles=self.config.n_particles,
            n_iterations=self.config.n_iterations,
        )

        return self.bpso_optimizer

    def _bpso_callback(
        self,
        state: dict[str, Any],
    ) -> None:
        self.last_bpso_state = dict(
            state
        )
        self.last_fitness_details = dict(
            state.get(
                "fitness_details",
                self.last_fitness_details,
            )
        )

        self._emit(
            "bpso_progress",
            **state,
        )

    # ------------------------------------------------------------------
    # Stage 5: BPSO run
    # ------------------------------------------------------------------

    def run_bpso(self) -> BPSOResult:
        if self.bpso_optimizer is None:
            self.initialize_bpso()

        assert self.bpso_optimizer is not None

        self.bpso_result = (
            self.bpso_optimizer.optimize()
        )

        self.selected_indices = [
            int(index)
            for index in np.flatnonzero(
                self.bpso_result.best_position
            )
        ]

        self.selected_features = [
            self.prepared.feature_names[index]
            for index in self.selected_indices
        ] if self.prepared is not None else []

        self.stage_status["bpso"] = "complete"

        reduction = feature_reduction(
            len(
                self.prepared.feature_names
            ),
            len(
                self.selected_features
            ),
        ) if self.prepared is not None else {}

        self._emit(
            "bpso_complete",
            result=self.bpso_result,
            selected_features=list(
                self.selected_features
            ),
            reduction=reduction,
        )

        return self.bpso_result

    # ------------------------------------------------------------------
    # Stage 6: selected-feature model
    # ------------------------------------------------------------------

    def train_selected_model(self) -> dict[str, Any]:
        if self.prepared is None:
            raise RuntimeError(
                "Prepare the dataset before training the selected model."
            )

        if self.bpso_result is None:
            self.run_bpso()

        if not self.selected_indices:
            raise RuntimeError(
                "BPSO selected no features."
            )

        self.selected_model = FixedNeuralNetwork(
            hidden_neurons=self.config.hidden_neurons,
            learning_rate=self.config.learning_rate,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            random_state=self.config.random_state,
        ).fit(
            self.prepared.X_train[
                :, self.selected_indices
            ],
            self.prepared.y_train,
        )

        validation = evaluate_model(
            self.selected_model,
            self.prepared.X_val[
                :, self.selected_indices
            ],
            self.prepared.y_val,
            feature_count=len(
                self.selected_indices
            ),
            labels=self.prepared.class_names,
        )

        test = evaluate_model(
            self.selected_model,
            self.prepared.X_test[
                :, self.selected_indices
            ],
            self.prepared.y_test,
            feature_count=len(
                self.selected_indices
            ),
            labels=self.prepared.class_names,
        )

        self.selected_metrics = {
            "validation": validation.to_dict(),
            "test": test.to_dict(),
            "model_summary": self.selected_model.summary(),
        }

        self.stage_status["selected_model"] = "complete"

        self._emit(
            "selected_model_trained",
            metrics=self.selected_metrics,
        )

        return self.selected_metrics

    def train_models_and_fuzzy(
        self,
        fuzzy_features: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Backward-compatible combined training/fuzzy entry point.

        Older tests and callers used ``train_models_and_fuzzy()`` after
        running BPSO. The current research architecture keeps the stages
        separate, so this wrapper preserves the old API while delegating to
        the new staged methods.
        """
        if self.baseline is None:
            self.train_baseline()

        return self.train_selected_and_fuzzy(
            fuzzy_features=fuzzy_features
        )

    # Compatibility alias used by older code.
    def train_selected_and_fuzzy(
        self,
        fuzzy_features: list[str] | None = None,
    ) -> dict[str, Any]:
        self.train_selected_model()
        if fuzzy_features is not None:
            self.configure_fuzzy(
                fuzzy_features
            )
        elif self.fuzzy_engine is None:
            try:
                self.configure_fuzzy()
            except ValueError:
                # A dataset with fewer than three selected features can still
                # complete NN evaluation; fuzzy mode remains unavailable.
                pass

        self.evaluate()
        return {
            "baseline_test": (
                self.baseline_metrics or {}
            ).get("test", {}),
            "selected_test": (
                self.selected_metrics or {}
            ).get("test", {}),
            "baseline_val": (
                self.baseline_metrics or {}
            ).get("validation", {}),
            "selected_val": (
                self.selected_metrics or {}
            ).get("validation", {}),
            "fuzzy_features": list(
                self.fuzzy_features
            ),
        }

    def _choose_fuzzy_features(
        self,
    ) -> list[str]:
        if self.prepared is None:
            raise RuntimeError(
                "Prepared data unavailable."
            )

        if len(self.selected_indices) < 3:
            raise ValueError(
                "At least three selected features are required "
                "for the current three-input fuzzy layer."
            )

        selected_matrix = self.prepared.X_train[
            :, self.selected_indices
        ]

        variances = np.var(
            selected_matrix,
            axis=0,
        )

        top_local = np.argsort(
            variances
        )[-3:][::-1]

        return [
            self.selected_features[index]
            for index in top_local
        ]

    # ------------------------------------------------------------------
    # Stage 7: fuzzy configuration / inference
    # ------------------------------------------------------------------

    def configure_fuzzy(
        self,
        feature_names: list[str] | None = None,
        *,
        config: FuzzyConfig | None = None,
    ) -> FuzzyRiskEngine:
        if self.prepared is None:
            raise RuntimeError(
                "Prepare the dataset before configuring fuzzy risk."
            )

        chosen = list(
            feature_names
            if feature_names is not None
            else self._choose_fuzzy_features()
        )

        if len(chosen) != 3:
            raise ValueError(
                "Exactly three fuzzy input features are required."
            )

        unknown = [
            name
            for name in chosen
            if name not in self.selected_features
        ]

        if unknown:
            raise ValueError(
                "Fuzzy input features must come from the BPSO-selected "
                f"feature subset: {unknown}"
            )

        self.fuzzy_features = list(
            chosen
        )

        self.fuzzy_indices = [
            self.prepared.feature_names.index(
                name
            )
            for name in self.fuzzy_features
        ]

        selected_train = self.prepared.X_train[
            :, self.fuzzy_indices
        ]

        self.fuzzy_min = np.min(
            selected_train,
            axis=0,
        )

        self.fuzzy_max = np.max(
            selected_train,
            axis=0,
        )

        if config is None:
            config = FuzzyConfig(
                feature_names=list(
                    self.fuzzy_features
                )
            )
        else:
            if config.feature_names != self.fuzzy_features:
                raise ValueError(
                    "FuzzyConfig feature_names must match the selected "
                    "fuzzy feature names."
                )

        self.fuzzy_engine = FuzzyRiskEngine(
            config
        )

        self.stage_status["fuzzy"] = "configured"

        self._emit(
            "fuzzy_configured",
            fuzzy_features=list(
                self.fuzzy_features
            ),
            fuzzy_config=config.to_dict(),
        )

        return self.fuzzy_engine

    def _normalize_fuzzy_values(
        self,
        transformed_rows: np.ndarray,
    ) -> np.ndarray:
        if (
            self.fuzzy_min is None
            or self.fuzzy_max is None
        ):
            raise RuntimeError(
                "Fuzzy normalization has not been initialized."
            )

        values = np.asarray(
            transformed_rows,
            dtype=float,
        )

        denominator = (
            self.fuzzy_max
            - self.fuzzy_min
        )

        denominator = np.where(
            np.abs(denominator) < 1e-12,
            1.0,
            denominator,
        )

        return np.clip(
            (
                values
                - self.fuzzy_min
            )
            / denominator,
            0.0,
            1.0,
        )

    def run_fuzzy(
        self,
        rows: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if self.fuzzy_engine is None:
            self.configure_fuzzy()

        if self.fuzzy_engine is None:
            raise RuntimeError(
                "Fuzzy engine is unavailable."
            )

        if rows is None:
            if self.prepared is None:
                raise RuntimeError(
                    "Prepared data is unavailable."
                )
            rows = self.prepared.X_test[
                :1,
                self.fuzzy_indices,
            ]
        else:
            rows = np.asarray(
                rows,
                dtype=float,
            )

        normalized = self._normalize_fuzzy_values(
            rows
        )

        results = [
            self.fuzzy_engine.infer(
                row
            )
            for row in normalized
        ]

        result = results[0]

        self.stage_status["fuzzy"] = "complete"

        self._emit(
            "fuzzy_complete",
            result=result,
        )

        return result

    # ------------------------------------------------------------------
    # Stage 8: final evaluation / comparison
    # ------------------------------------------------------------------

    def evaluate(self) -> dict[str, Any]:
        if self.prepared is None:
            raise RuntimeError(
                "Prepare the dataset before evaluation."
            )

        if self.baseline is None:
            self.train_baseline()

        if self.selected_model is None:
            self.train_selected_model()

        assert self.baseline is not None
        assert self.selected_model is not None

        self.comparison = compare_models(
            self.baseline,
            self.selected_model,
            self.prepared.X_test,
            self.prepared.X_test[
                :, self.selected_indices
            ],
            self.prepared.y_test,
            original_features=len(
                self.prepared.feature_names
            ),
            selected_features=len(
                self.selected_features
            ),
            class_names=self.prepared.class_names,
        )

        self.baseline_metrics = {
            "validation": self.baseline_metrics["validation"]
            if self.baseline_metrics
            else {},
            "test": self.comparison.baseline.to_dict(),
            "model_summary": self.baseline.summary(),
        }

        self.selected_metrics = {
            "validation": self.selected_metrics["validation"]
            if self.selected_metrics
            else {},
            "test": self.comparison.bpso.to_dict(),
            "model_summary": self.selected_model.summary(),
        }

        self.stage_status["evaluation"] = "complete"

        result = self.comparison.to_dict()

        self._emit(
            "evaluation_complete",
            comparison=result,
        )

        return result

    # ------------------------------------------------------------------
    # Stage 9: prediction
    # ------------------------------------------------------------------

    def predict_record(
        self,
        row: pd.DataFrame,
    ) -> dict[str, Any]:
        if self.selected_model is None:
            self.load_artifacts()

        if self.selected_model is None:
            raise RuntimeError(
                "Selected model is not available."
            )

        if self.prepared is None:
            raise RuntimeError(
                "Prepared data is unavailable."
            )

        transformed = self.preprocessor.transform_dataframe(
            row
        )

        selected_x = transformed[
            :,
            self.selected_indices,
        ]

        prediction, confidence = (
            self.selected_model.predict_with_confidence(
                selected_x
            )
        )

        encoded_prediction = int(
            np.asarray(
                prediction
            )[0]
        )

        prediction_label = self.prepared.class_names[
            encoded_prediction
        ]

        result: dict[str, Any] = {
            "prediction": prediction_label,
            "prediction_encoded": encoded_prediction,
            "confidence": float(
                confidence[0]
            ),
            "selected_features": list(
                self.selected_features
            ),
        }

        if self.fuzzy_engine is not None:
            fuzzy_transformed = transformed[
                :, self.fuzzy_indices
            ]
            fuzzy_normalized = self._normalize_fuzzy_values(
                fuzzy_transformed
            )
            fuzzy_result = self.fuzzy_engine.infer(
                fuzzy_normalized[0]
            )
            fuzzy_result["explanation"] = (
                self.fuzzy_engine.explain(
                    fuzzy_result
                )
            )
            result["fuzzy"] = fuzzy_result
        else:
            result["fuzzy"] = None

        return result

    # ------------------------------------------------------------------
    # Stage 10: artifacts / report
    # ------------------------------------------------------------------

    def _artifact_dir(self) -> Path:
        path = Path(
            self.config.artifacts_dir
        )
        path.mkdir(
            parents=True,
            exist_ok=True,
        )
        return path

    def _serializable_config(self) -> dict[str, Any]:
        return self.config.to_dict()

    def build_results_payload(
        self,
        experiment_seconds: float | None = None,
    ) -> dict[str, Any]:
        if self.prepared is None:
            raise RuntimeError(
                "No prepared dataset is available."
            )

        reduction = feature_reduction(
            len(self.prepared.feature_names),
            len(self.selected_features),
        )

        payload: dict[str, Any] = {
            "mode": getattr(self.config, "mode", "research"),
            "target_column": self.prepared.target_column,
            "class_names": list(
                self.prepared.class_names
            ),
            "total_features": len(
                self.prepared.feature_names
            ),
            "selected_features": list(
                self.selected_features
            ),
            "selected_feature_count": len(
                self.selected_features
            ),
            "removed_features": reduction[
                "removed_features"
            ],
            "feature_reduction_percent": reduction[
                "reduction_percent"
            ],
            "bpso": {
                "best_fitness": (
                    self.bpso_result.best_fitness
                    if self.bpso_result is not None
                    else None
                ),
                "best_details": (
                    self.bpso_result.best_details
                    if self.bpso_result is not None
                    else {}
                ),
                "history": (
                    self.bpso_result.history
                    if self.bpso_result is not None
                    else []
                ),
            },
            "baseline": self.baseline_metrics,
            "selected_model": self.selected_metrics,
            "metrics": {
                "baseline_test": (
                    self.baseline_metrics or {}
                ).get("test", {}),
                "selected_test": (
                    self.selected_metrics or {}
                ).get("test", {}),
                "baseline_val": (
                    self.baseline_metrics or {}
                ).get("validation", {}),
                "selected_val": (
                    self.selected_metrics or {}
                ).get("validation", {}),
                "fuzzy_features": list(
                    self.fuzzy_features
                ),
            },
            "comparison": (
                self.comparison.to_dict()
                if self.comparison is not None
                else None
            ),
            "fuzzy": {
                "features": list(
                    self.fuzzy_features
                ),
                "config": (
                    self.fuzzy_engine.config.to_dict()
                    if self.fuzzy_engine is not None
                    else None
                ),
            },
            "config": self._serializable_config(),
        }

        if experiment_seconds is not None:
            payload[
                "experiment_seconds"
            ] = float(experiment_seconds)

        return payload

    def save_artifacts(
        self,
        metrics: dict[str, Any] | None = None,
    ) -> Path:
        if self.prepared is None:
            raise RuntimeError(
                "No prepared data available."
            )

        out = self._artifact_dir()

        self.preprocessor.save(
            out / "preprocessor.joblib"
        )

        if self.baseline is not None:
            self.baseline.save(
                out / "baseline_model.joblib"
            )

        if self.selected_model is not None:
            self.selected_model.save(
                out / "selected_model.joblib"
            )

        if self.bpso_result is not None:
            pd.DataFrame(
                self.bpso_result.history
            ).to_csv(
                out / "bpso_history.csv",
                index=False,
            )

            (out / "selected_features.json").write_text(
                json.dumps(
                    {
                        "feature_names": self.selected_features,
                        "indices": self.selected_indices,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        if self.fuzzy_engine is not None:
            (out / "fuzzy_config.json").write_text(
                json.dumps(
                    self.fuzzy_engine.config.to_dict(),
                    indent=2,
                ),
                encoding="utf-8",
            )

            if (
                self.fuzzy_min is not None
                and self.fuzzy_max is not None
            ):
                np.savez(
                    out / "fuzzy_scaler.npz",
                    minimum=self.fuzzy_min,
                    maximum=self.fuzzy_max,
                )

        payload = self.build_results_payload()

        if metrics is not None:
            payload["legacy_metrics"] = metrics

        (out / "results.json").write_text(
            json.dumps(
                payload,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        self.stage_status["report"] = "complete"

        return out

    def export_report(
        self,
        output_path: str | Path | None = None,
    ) -> Path:
        payload = self.build_results_payload()

        output = Path(
            output_path
            if output_path is not None
            else self._artifact_dir()
            / "experiment_report.json"
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output.write_text(
            json.dumps(
                payload,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        self.stage_status["report"] = "complete"

        return output

    def load_artifacts(
        self,
        artifact_dir: str | Path | None = None,
    ) -> None:
        out = Path(
            artifact_dir
            if artifact_dir is not None
            else self.config.artifacts_dir
        )

        preprocessor_path = (
            out / "preprocessor.joblib"
        )
        selected_model_path = (
            out / "selected_model.joblib"
        )
        baseline_model_path = (
            out / "baseline_model.joblib"
        )

        self.preprocessor = Preprocessor.load(
            preprocessor_path
        )

        self.prepared = None
        self.baseline = (
            FixedNeuralNetwork.load(
                baseline_model_path
            )
            if baseline_model_path.exists()
            else None
        )

        self.selected_model = (
            FixedNeuralNetwork.load(
                selected_model_path
            )
            if selected_model_path.exists()
            else None
        )

        selected_features_path = (
            out / "selected_features.json"
        )

        if selected_features_path.exists():
            selected_payload = json.loads(
                selected_features_path.read_text(
                    encoding="utf-8"
                )
            )
            self.selected_features = list(
                selected_payload.get(
                    "feature_names",
                    [],
                )
            )
            self.selected_indices = [
                int(value)
                for value in selected_payload.get(
                    "indices",
                    [],
                )
            ]

        fuzzy_config_path = (
            out / "fuzzy_config.json"
        )

        if fuzzy_config_path.exists():
            fuzzy_config = FuzzyConfig.from_dict(
                json.loads(
                    fuzzy_config_path.read_text(
                        encoding="utf-8"
                    )
                )
            )
            self.fuzzy_features = list(
                fuzzy_config.feature_names
            )
            self.fuzzy_indices = [
                self.preprocessor.feature_names.index(
                    name
                )
                for name in self.fuzzy_features
            ]
            self.fuzzy_engine = FuzzyRiskEngine(
                fuzzy_config
            )

        fuzzy_scaler_path = (
            out / "fuzzy_scaler.npz"
        )

        if fuzzy_scaler_path.exists():
            values = np.load(
                fuzzy_scaler_path
            )
            self.fuzzy_min = values["minimum"]
            self.fuzzy_max = values["maximum"]

        results_path = (
            out / "results.json"
        )

        if results_path.exists():
            payload = json.loads(
                results_path.read_text(
                    encoding="utf-8"
                )
            )
            self.dataset_profile = {
                "target_column": payload.get(
                    "target_column"
                ),
                "usable_features": payload.get(
                    "total_features"
                ),
                "classes": payload.get(
                    "class_names",
                    [],
                ),
            }

    # ------------------------------------------------------------------
    # Complete experiment
    # ------------------------------------------------------------------

    def run_full_experiment(
        self,
        fuzzy_features: list[str] | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()

        self._emit(
            "experiment_started"
        )

        if self.raw_dataframe is None:
            raise RuntimeError(
                "Load a dataframe before running an experiment."
            )

        if self.prepared is None:
            self.prepare_dataset()

        self.train_baseline()
        self.initialize_bpso()
        self.run_bpso()
        self.train_selected_model()

        # Fuzzy mode is expected to have three BPSO-selected numeric inputs.
        # A dataset with too few selected features can still finish the NN
        # comparison; the fuzzy stage is then reported as unavailable.
        try:
            self.configure_fuzzy(
                fuzzy_features
            )
            self.run_fuzzy()
        except ValueError as exc:
            self.fuzzy_engine = None
            self.stage_status["fuzzy"] = (
                f"unavailable: {exc}"
            )

        self.evaluate()

        elapsed = (
            time.perf_counter()
            - start
        )

        payload = self.build_results_payload(
            experiment_seconds=elapsed
        )

        self.save_artifacts()

        self._emit(
            "experiment_complete",
            results=payload,
        )

        return payload

    # Compatibility alias.
    def run(self) -> dict[str, Any]:
        return self.run_full_experiment()
