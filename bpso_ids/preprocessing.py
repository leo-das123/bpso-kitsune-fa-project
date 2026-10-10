from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.pipeline import Pipeline

from .schema import (
    detect_datetime_columns,
    detect_id_candidates,
    detect_metadata_columns,
    detect_target_candidates,
)


@dataclass
class PreparedData:
    """
    Fully transformed train/validation/test data used by the models.
    """

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray

    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray

    feature_names: list[str]

    target_column: str

    class_names: list[str]


class Preprocessor:
    """
    Leakage-safe generic tabular preprocessing for supervised classification.

    Main workflow:

        raw dataframe
            ↓
        target selection
            ↓
        ID / metadata removal
            ↓
        duplicate removal
            ↓
        invalid numeric value normalization
            ↓
        train / validation / test split
            ↓
        fit imputers / encoders / scaler on train only
            ↓
        transform validation/test

    This class intentionally supports both the new generic API and the
    existing pipeline API.
    """

    def __init__(
        self,
        random_state: int = 42,
        test_size: float = 0.15,
        validation_size: float = 0.15,
        max_classes: int = 20,
    ) -> None:
        if not 0 < test_size < 1:
            raise ValueError(
                "test_size must be between 0 and 1."
            )

        if not 0 < validation_size < 1:
            raise ValueError(
                "validation_size must be between 0 and 1."
            )

        if test_size + validation_size >= 1:
            raise ValueError(
                "test_size + validation_size must be < 1."
            )

        if max_classes < 2:
            raise ValueError(
                "max_classes must be >= 2."
            )

        self.random_state = int(
            random_state
        )

        self.test_size = float(
            test_size
        )

        self.validation_size = float(
            validation_size
        )

        self.max_classes = int(
            max_classes
        )

        self.transformer: ColumnTransformer | None = None

        self.label_encoder = LabelEncoder()

        self.feature_names: list[str] = []

        self.raw_feature_columns: list[str] = []

        self.numeric_columns: list[str] = []

        self.categorical_columns: list[str] = []

        self.datetime_columns: list[str] = []

        self.target_column: str | None = None

        self.dropped_columns: list[str] = []

        self.constant_columns: list[str] = []

        self._prepared: PreparedData | None = None

        self._fit_drop_columns: list[str] = []

    # ------------------------------------------------------------------
    # Generic cleaning helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dataframe(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        if not isinstance(
            df,
            pd.DataFrame,
        ):
            raise TypeError(
                "df must be a pandas DataFrame."
            )

        if df.empty:
            raise ValueError(
                "Dataset is empty."
            )

        out = df.copy()

        out.columns = [
            str(column)
            for column in out.columns
        ]

        # Normalize infinity to missing.
        out = out.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        # Duplicate removal belongs to preprocessing.
        out = (
            out
            .drop_duplicates()
            .reset_index(drop=True)
        )

        return out

    @staticmethod
    def _make_one_hot_encoder() -> OneHotEncoder:
        """
        Support both newer and older scikit-learn releases.
        """
        try:
            return OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )
        except TypeError:
            # Older scikit-learn.
            return OneHotEncoder(
                handle_unknown="ignore",
                sparse=False,
            )

    @staticmethod
    def _make_pipeline(
        steps: list[tuple[str, object]],
    ) -> Pipeline:
        return Pipeline(steps)

    def _validate_target(
        self,
        y: pd.Series,
    ) -> None:
        unique_count = int(
            y.nunique(
                dropna=True
            )
        )

        if unique_count < 2:
            raise ValueError(
                "The target must contain at least "
                "two classes for supervised classification."
            )

        if unique_count > self.max_classes:
            raise ValueError(
                f"The target contains {unique_count} "
                f"unique classes, exceeding the configured "
                f"maximum of {self.max_classes}. "
                "Select a categorical classification target "
                "explicitly."
            )

    def _sample_rows(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        max_rows: int | None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Deterministically downsample large datasets.

        Stratified sampling is preferred; random sampling is the fallback.
        """
        if (
            max_rows is None
            or len(X) <= max_rows
        ):
            return (
                X.reset_index(drop=True),
                y.reset_index(drop=True),
            )

        if max_rows < 10:
            raise ValueError(
                "max_rows must be >= 10 when used."
            )

        try:
            X_sample, _, y_sample, _ = train_test_split(
                X,
                y,
                train_size=max_rows,
                stratify=y,
                random_state=self.random_state,
            )

            return (
                X_sample.reset_index(drop=True),
                y_sample.reset_index(drop=True),
            )

        except ValueError:
            rng = np.random.default_rng(
                self.random_state
            )

            indices = rng.choice(
                len(X),
                size=max_rows,
                replace=False,
            )

            return (
                X.iloc[indices].reset_index(drop=True),
                y.iloc[indices].reset_index(drop=True),
            )

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        target_column: str,
        drop_columns: Iterable[str] | None = None,
        max_rows: int | None = None,
    ) -> "Preprocessor":
        """
        New generic API.

        Example:

            preprocessor.fit(
                df,
                target_column="attack",
                drop_columns=["record_id"],
            )

        After fitting:

            prepared = preprocessor.transform_and_split()
        """
        self.fit_prepare(
            df=df,
            target_column=target_column,
            drop_columns=drop_columns,
            max_rows=max_rows,
        )

        return self

    def prepare(
        self,
        df: pd.DataFrame,
        target_column: str | None = None,
        drop_columns: Iterable[str] | None = None,
        max_rows: int | None = None,
    ) -> PreparedData:
        """
        Backward-compatible convenience wrapper.

        Older application/test code used:

            preprocessor.prepare(df)

        The newer pipeline uses:

            preprocessor.fit_prepare(
                df,
                target_column=...
            )

        This method keeps both APIs working.
        """

        clean = self._to_dataframe(df)

        # --------------------------------------------------------------
        # Resolve target automatically when the caller does not provide it.
        # --------------------------------------------------------------
        if target_column is None:
            candidates = detect_target_candidates(clean)

            if not candidates:
                raise ValueError(
                    "Unable to automatically determine the target column. "
                    "Please provide target_column explicitly."
                )

            target_column = candidates[0]

        # --------------------------------------------------------------
        # Detect metadata columns such as Scenario, Source, File, etc.
        # These are not predictive features in the generic workflow.
        # --------------------------------------------------------------
        automatic_metadata = set(
            detect_metadata_columns(
                clean,
                exclude={target_column},
            )
        )

        requested_drop = set(
            str(column)
            for column in (drop_columns or [])
        )

        requested_drop.update(
            automatic_metadata
        )

        requested_drop.discard(
            str(target_column)
        )

        return self.fit_prepare(
            clean,
            target_column=str(target_column),
            drop_columns=sorted(requested_drop),
            max_rows=max_rows,
        )

    def transform_and_split(
        self,
        df: pd.DataFrame | None = None,
        target_column: str | None = None,
        drop_columns: Iterable[str] | None = None,
        max_rows: int | None = None,
    ) -> PreparedData:
        """
        Return the prepared train/validation/test data.

        If df is supplied, this method refits the preprocessing pipeline.
        Without df, it returns the result from the most recent fit/prepare call.
        """
        if df is not None:
            if target_column is None:
                if self.target_column is None:
                    raise ValueError(
                        "target_column must be supplied "
                        "when fitting from a new dataframe."
                    )

                target_column = self.target_column

            return self.fit_prepare(
                df=df,
                target_column=target_column,
                drop_columns=drop_columns,
                max_rows=max_rows,
            )

        if self._prepared is None:
            raise RuntimeError(
                "Preprocessor has not been fitted."
            )

        return self._prepared

    def fit_prepare(
        self,
        df: pd.DataFrame,
        target_column: str,
        drop_columns: Iterable[str] | None = None,
        max_rows: int | None = None,
    ) -> PreparedData:
        """
        Backward-compatible full preprocessing method used by the current
        ResearchPipeline.
        """
        out = self._to_dataframe(df)

        target_column = str(
            target_column
        )

        if target_column not in out.columns:
            raise ValueError(
                f"Target column not found: "
                f"{target_column}"
            )

        self.target_column = target_column

        requested_drop = {
            str(column)
            for column in (
                drop_columns or []
            )
        }

        # Never allow callers to accidentally remove the target here.
        requested_drop.discard(
            target_column
        )

        # Conservative automatic ID detection.
        auto_ids = set(
            detect_id_candidates(
                out,
                exclude={target_column},
            )
        )

        # Datetime metadata is not directly fed to the current tabular
        # classifier. We report it as dropped rather than silently encoding it.
        datetime_candidates = detect_datetime_columns(
            out,
            exclude={
                target_column
            }
            | auto_ids
            | requested_drop,
        )

        all_drop = (
            requested_drop
            | auto_ids
            | set(datetime_candidates)
        )

        self.dropped_columns = sorted(
            str(column)
            for column in all_drop
        )

        self.datetime_columns = sorted(
            str(column)
            for column in datetime_candidates
        )

        self._fit_drop_columns = list(
            self.dropped_columns
        )

        # Separate the target before feature processing.
        y_raw = out[target_column].copy()

        feature_df = out.drop(
            columns=[
                target_column
            ]
            + self.dropped_columns,
            errors="ignore",
        ).copy()

        if feature_df.shape[1] == 0:
            raise ValueError(
                "No candidate feature columns remain "
                "after removing metadata/ID columns."
            )

        # Target rows with missing labels cannot be used for supervised learning.
        valid_target = y_raw.notna()

        feature_df = (
            feature_df
            .loc[valid_target]
            .reset_index(drop=True)
        )

        y_raw = (
            y_raw
            .loc[valid_target]
            .reset_index(drop=True)
        )

        self._validate_target(
            y_raw
        )

        # Reduce development data size when requested.
        feature_df, y_raw = self._sample_rows(
            feature_df,
            y_raw,
            max_rows,
        )

        # Remove constant features after sampling.
        constant_columns = [
            str(column)
            for column in feature_df.columns
            if feature_df[column].nunique(
                dropna=False
            ) <= 1
        ]

        self.constant_columns = sorted(
            constant_columns
        )

        if constant_columns:
            feature_df = feature_df.drop(
                columns=constant_columns
            )

            self.dropped_columns = sorted(
                set(self.dropped_columns)
                | set(constant_columns)
            )

        if feature_df.shape[1] == 0:
            raise ValueError(
                "No usable candidate features remain "
                "after removing constant columns."
            )

        self.raw_feature_columns = [
            str(column)
            for column in feature_df.columns
        ]

        self.numeric_columns = [
            str(column)
            for column in feature_df.select_dtypes(
                include=[np.number]
            ).columns
        ]

        self.categorical_columns = [
            str(column)
            for column in feature_df.columns
            if str(column)
            not in set(self.numeric_columns)
        ]

        # ------------------------------------------------------------------
        # Leakage-safe split FIRST.
        # ------------------------------------------------------------------

        y_as_string = y_raw.astype(
            str
        )

        y_encoded = (
            self.label_encoder
            .fit_transform(y_as_string)
        )

        class_counts = pd.Series(
            y_encoded
        ).value_counts()

        can_stratify = (
            len(class_counts) >= 2
            and int(class_counts.min()) >= 2
        )

        X_train_df, X_temp_df, y_train_enc, y_temp_enc = (
            self._safe_train_test_split(
                feature_df,
                y_encoded,
                test_size=self._test_plus_validation(),
                stratify=(
                    y_encoded
                    if can_stratify
                    else None
                ),
            )
        )

        val_fraction_of_temp = (
            self._validation_fraction_of_temp()
        )

        try:
            temp_class_counts = (
                pd.Series(
                    y_temp_enc
                ).value_counts()
            )

            can_stratify_temp = (
                len(temp_class_counts) >= 2
                and int(
                    temp_class_counts.min()
                ) >= 2
            )

        except Exception:
            can_stratify_temp = False

        X_val_df, X_test_df, y_val, y_test = (
            self._safe_train_test_split(
                X_temp_df,
                y_temp_enc,
                test_size=1.0 - val_fraction_of_temp,
                stratify=(
                    y_temp_enc
                    if can_stratify_temp
                    else None
                ),
            )
        )

        # ------------------------------------------------------------------
        # Fit transformations using TRAINING data only.
        # ------------------------------------------------------------------

        numeric_pipeline = self._make_pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        categorical_pipeline = self._make_pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "encoder",
                    self._make_one_hot_encoder(),
                ),
            ]
        )

        transformers = []

        if self.numeric_columns:
            transformers.append(
                (
                    "num",
                    numeric_pipeline,
                    self.numeric_columns,
                )
            )

        if self.categorical_columns:
            transformers.append(
                (
                    "cat",
                    categorical_pipeline,
                    self.categorical_columns,
                )
            )

        if not transformers:
            raise ValueError(
                "No numeric or categorical feature columns remain."
            )

        self.transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )

        self.transformer.fit(
            X_train_df
        )

        X_train = np.asarray(
            self.transformer.transform(
                X_train_df
            ),
            dtype=float,
        )

        X_val = np.asarray(
            self.transformer.transform(
                X_val_df
            ),
            dtype=float,
        )

        X_test = np.asarray(
            self.transformer.transform(
                X_test_df
            ),
            dtype=float,
        )

        y_train = np.asarray(
            y_train_enc,
            dtype=int,
        )

        y_val = np.asarray(
            y_val,
            dtype=int,
        )

        y_test = np.asarray(
            y_test,
            dtype=int,
        )

        self.feature_names = [
            str(name)
            for name in self.transformer.get_feature_names_out()
        ]

        self._prepared = PreparedData(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            feature_names=list(
                self.feature_names
            ),
            target_column=target_column,
            class_names=[
                str(name)
                for name in self.label_encoder.classes_
            ],
        )

        return self._prepared

    # ------------------------------------------------------------------
    # Split helpers
    # ------------------------------------------------------------------

    def _safe_train_test_split(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        *,
        test_size: float,
        stratify: np.ndarray | None,
    ):
        try:
            return train_test_split(
                X,
                y,
                test_size=test_size,
                stratify=stratify,
                random_state=self.random_state,
            )

        except ValueError:
            return train_test_split(
                X,
                y,
                test_size=test_size,
                stratify=None,
                random_state=self.random_state,
            )

    @property
    def train_fraction(self) -> float:
        return (
            1.0
            - self.validation_size
            - self.test_size
        )

    def _test_plus_validation(self) -> float:
        return (
            self.test_size
            + self.validation_size
        )

    def _validation_fraction_of_temp(
        self,
    ) -> float:
        return (
            self.validation_size
            / self._test_plus_validation()
        )

    # ------------------------------------------------------------------
    # Inference / serialization
    # ------------------------------------------------------------------

    def transform_dataframe(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Transform new records with the already-fitted preprocessing pipeline.

        The input may contain:
          - raw feature columns
          - extra ID/metadata columns
          - the original target column

        Only the fitted raw feature columns are used.
        """
        if self.transformer is None:
            raise RuntimeError(
                "Preprocessor has not been fitted."
            )

        clean = df.copy()

        clean.columns = [
            str(column)
            for column in clean.columns
        ]

        missing = [
            column
            for column in self.raw_feature_columns
            if column not in clean.columns
        ]

        if missing:
            raise ValueError(
                "Missing raw feature columns: "
                f"{missing[:10]}"
            )

        clean = clean[
            self.raw_feature_columns
        ].copy()

        transformed = (
            self.transformer.transform(
                clean
            )
        )

        return np.asarray(
            transformed,
            dtype=float,
        )

    def encode_labels(
        self,
        values: Iterable[object],
    ) -> np.ndarray:
        return self.label_encoder.transform(
            [str(value) for value in values]
        )

    def decode_labels(
        self,
        y: np.ndarray,
    ) -> list[str]:
        return [
            str(value)
            for value in self.label_encoder.inverse_transform(
                np.asarray(
                    y,
                    dtype=int,
                )
            )
        ]

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
    ) -> "Preprocessor":
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Preprocessor artifact not found: {path}"
            )

        return joblib.load(path)