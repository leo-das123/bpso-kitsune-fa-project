from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

from bpso_ids.config import ProjectConfig
from bpso_ids.pipeline import ResearchPipeline


class ExperimentWorkflow:
    """
    Application-facing bridge between NiceGUI and the research pipeline.

    NiceGUI should communicate with this class instead of directly calling
    lower-level algorithm classes such as BPSO, Neural Network, or Fuzzy.

    Architecture:

        NiceGUI
            ↓
        ExperimentWorkflow
            ↓
        ResearchPipeline
            ↓
        BPSO / Neural Network / Fuzzy
    """

    def __init__(
        self,
        config: ProjectConfig,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.pipeline = ResearchPipeline(
            config,
            progress_callback=progress_callback,
        )

    # ==============================================================
    # Properties
    # ==============================================================

    @property
    def config(self) -> ProjectConfig:
        """
        Return the active project configuration.
        """
        return self.pipeline.config

    @property
    def dataset_profile(self) -> dict[str, Any] | None:
        """
        Return the analyzed dataset profile.
        """
        return self.pipeline.dataset_profile

    @property
    def selected_features(self) -> list[str]:
        """
        Return the features selected by BPSO.
        """
        return list(self.pipeline.selected_features)

    @property
    def stage_status(self) -> dict[str, str]:
        """
        Return the current status of every research pipeline stage.
        """
        return dict(self.pipeline.stage_status)

    # ==============================================================
    # Progress callback
    # ==============================================================

    def set_progress_callback(
        self,
        callback: Callable[[dict[str, Any]], None] | None,
    ) -> None:
        """
        Attach or replace the progress callback.

        NiceGUI can use this later to update:
        - progress bars
        - iteration counters
        - convergence charts
        - status messages
        """
        self.pipeline.progress_callback = callback

    # ==============================================================
    # Dataset
    # ==============================================================

    def load_csv(
        self,
        path: str | Path,
        target_column: str | None = None,
        drop_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Load a CSV dataset and perform initial schema analysis.

        This method intentionally does NOT run preprocessing or training.
        Those are separate visible stages for the UI.
        """
        df = pd.read_csv(path)

        return self.load_dataframe(
            df,
            target_column=target_column,
            drop_columns=drop_columns,
        )

    def load_dataframe(
        self,
        df: pd.DataFrame,
        target_column: str | None = None,
        drop_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Load an already-created DataFrame and analyze its schema.

        Parameters
        ----------
        df:
            Input pandas DataFrame.

        target_column:
            Optional supervised-learning target column.

        drop_columns:
            Optional metadata columns that should later be excluded.

        Returns
        -------
        dict
            Dataset profile.
        """
        self.pipeline.analyze_dataset(
            df,
            target_column=target_column,
        )

        return self.pipeline.dataset_profile or {}

    def analyze_dataset(
        self,
        target_column: str | None = None,
    ) -> dict[str, Any]:
        """
        Re-run dataset analysis on the currently loaded DataFrame.
        """
        if self.pipeline.raw_dataframe is None:
            raise RuntimeError(
                "No dataframe has been loaded."
            )

        return self.pipeline.analyze_dataset(
            self.pipeline.raw_dataframe,
            target_column=target_column,
        )

    # ==============================================================
    # Preprocessing
    # ==============================================================

    def prepare_dataset(
        self,
        target_column: str | None = None,
        drop_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Run the preprocessing stage.

        The ResearchPipeline handles:
        - target resolution
        - ID detection
        - metadata removal
        - missing-value handling
        - categorical encoding
        - scaling
        - train/validation/test splitting

        Returns a compact UI-friendly summary.
        """
        prepared = self.pipeline.prepare_dataset(
            target_column=target_column,
            drop_columns=drop_columns,
        )

        return {
            "target_column": prepared.target_column,
            "train_rows": len(
                prepared.X_train
            ),
            "validation_rows": len(
                prepared.X_val
            ),
            "test_rows": len(
                prepared.X_test
            ),
            "features": len(
                prepared.feature_names
            ),
            "feature_names": list(
                prepared.feature_names
            ),
            "classes": list(
                prepared.class_names
            ),
        }

    # ==============================================================
    # Baseline Neural Network
    # ==============================================================

    def train_baseline(self) -> dict[str, Any]:
        """
        Train the baseline neural network using all prepared features.
        """
        return self.pipeline.train_baseline()

    # ==============================================================
    # BPSO
    # ==============================================================

    def initialize_bpso(self) -> dict[str, Any]:
        """
        Initialize the Binary Particle Swarm Optimization stage.

        Returns optimizer information useful for the UI.
        """
        optimizer = self.pipeline.initialize_bpso()

        return {
            "n_features": optimizer.n_features,
            "n_particles": optimizer.n_particles,
            "n_iterations": optimizer.n_iterations,
            "min_selected_features": optimizer.min_selected_features,
        }

    def run_bpso(self) -> dict[str, Any]:
        """
        Execute BPSO feature selection.

        Returns a UI-friendly result containing:
        - best fitness
        - fitness details
        - selected features
        - selected indices
        - convergence history
        - final particle state
        """
        result = self.pipeline.run_bpso()

        return {
            "best_fitness": float(
                result.best_fitness
            ),
            "best_details": dict(
                result.best_details
            ),
            "selected_features": list(
                self.pipeline.selected_features
            ),
            "selected_indices": list(
                self.pipeline.selected_indices
            ),
            "history": list(
                result.history
            ),
            "last_state": dict(
                result.final_state
            ),
        }

    # ==============================================================
    # Selected Feature Neural Network
    # ==============================================================

    def train_selected_model(self) -> dict[str, Any]:
        """
        Train the neural network again using only BPSO-selected features.
        """
        return self.pipeline.train_selected_model()

    # ==============================================================
    # Fuzzy Risk Assessment
    # ==============================================================

    def configure_fuzzy(
        self,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Configure the fuzzy risk layer.

        If feature_names are not supplied, the pipeline chooses appropriate
        features from the selected feature set.
        """
        engine = self.pipeline.configure_fuzzy(
            feature_names
        )

        return engine.config.to_dict()

    def run_fuzzy(self) -> dict[str, Any]:
        """
        Execute the fuzzy risk assessment stage.
        """
        return self.pipeline.run_fuzzy()

    # ==============================================================
    # Evaluation / Comparison
    # ==============================================================

    def evaluate(self) -> dict[str, Any]:
        """
        Evaluate baseline vs BPSO-selected model.
        """
        return self.pipeline.evaluate()

    # ==============================================================
    # Complete experiment
    # ==============================================================

    def run_all(
        self,
        fuzzy_features: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Execute the complete research experiment.

        Pipeline:

            Load
              ↓
            Preprocess
              ↓
            Baseline NN
              ↓
            BPSO
              ↓
            Selected NN
              ↓
            Fuzzy Risk
              ↓
            Evaluation
              ↓
            Artifacts / Report
        """
        return self.pipeline.run_full_experiment(
            fuzzy_features=fuzzy_features
        )

    # ==============================================================
    # Report
    # ==============================================================

    def export_report(
        self,
        path: str | Path | None = None,
    ) -> str:
        """
        Export the experiment report.

        Returns the final report path as a string so NiceGUI can easily
        display it or expose it through a download control.
        """
        return str(
            self.pipeline.export_report(
                path
            )
        )

    # ==============================================================
    # Single-record analysis
    # ==============================================================

    def predict(
        self,
        row: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Run prediction and fuzzy risk analysis for a single record.

        This will later power the:
            "Single Traffic Analysis"
        page in NiceGUI.
        """
        return self.pipeline.predict_record(
            row
        )

    # ==============================================================
    # Artifact loading
    # ==============================================================

    def load_artifacts(
        self,
        artifact_dir: str | Path | None = None,
    ) -> None:
        """
        Load previously trained artifacts.

        This allows the UI to reopen an existing experiment without
        retraining the models.
        """
        self.pipeline.load_artifacts(
            artifact_dir
        )