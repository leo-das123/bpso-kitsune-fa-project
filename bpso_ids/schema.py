from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable
import re

import pandas as pd


# ---------------------------------------------------------------------------
# Heuristics used by the generic CSV analyzer.
# These are intentionally conservative: the application should suggest
# targets / IDs, not silently delete meaningful user features.
# ---------------------------------------------------------------------------

TARGET_PATTERNS = (
    re.compile(r"^(label|target|class|y)$", re.IGNORECASE),
    re.compile(
        r"(label|target|class|attack|anomaly|outcome|result|"
        r"response|diagnosis|disease|status|category)$",
        re.IGNORECASE,
    ),
)

ID_PATTERNS = re.compile(
    r"(^id$|_id$|^index$|^idx$|record_id|row_id|"
    r"sample_id|event_id|uuid|guid)$",
    re.IGNORECASE,
)

METADATA_PATTERNS = re.compile(
    r"^(scenario|source_file|source|filename|"
    r"file|dataset|partition|split)$",
    re.IGNORECASE,
)


@dataclass
class DatasetProfile:
    """Machine-readable summary of a tabular dataset."""

    rows: int
    columns: int

    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    datetime_columns: list[str] = field(default_factory=list)

    constant_columns: list[str] = field(default_factory=list)

    possible_id_columns: list[str] = field(default_factory=list)
    possible_target_columns: list[str] = field(default_factory=list)

    missing_values: dict[str, int] = field(default_factory=dict)
    duplicate_rows: int = 0

    class_distribution: dict[str, int] = field(default_factory=dict)

    target_column: str | None = None
    dropped_columns: list[str] = field(default_factory=list)

    # Helpful summary values for UI/reporting.
    total_missing_cells: int = 0
    numeric_count: int = 0
    categorical_count: int = 0
    datetime_count: int = 0
    constant_count: int = 0
    target_class_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def has_missing_values(self) -> bool:
        return self.total_missing_cells > 0

    @property
    def is_binary_target(self) -> bool:
        return self.target_class_count == 2

    @property
    def is_multiclass_target(self) -> bool:
        return self.target_class_count > 2


def _safe_name_list(columns: Iterable[object]) -> list[str]:
    return [str(column) for column in columns]


def _is_monotonic_integer_index(series: pd.Series) -> bool:
    """
    Detect columns such as:
        0, 1, 2, 3, ...
    or
        1, 2, 3, 4, ...

    This is intentionally stricter than simply checking uniqueness.
    A unique continuous measurement should not automatically be treated
    as an ID column.
    """
    if len(series) < 2:
        return False

    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().mean() < 1.0:
        return False

    values = numeric.to_numpy()

    if not pd.Series(values).is_monotonic_increasing:
        return False

    zero_based = list(range(len(values)))
    one_based = list(range(1, len(values) + 1))

    as_int = [int(value) for value in values]

    return as_int == zero_based or as_int == one_based


def _target_score(
    df: pd.DataFrame,
    column: str,
) -> tuple[int, int, float, str]:
    """
    Rank target candidates.

    Higher:
      1. name match
      2. classification-like cardinality
      3. smaller number of unique classes
    """
    series = df[column]
    unique_count = int(series.nunique(dropna=True))
    row_count = max(len(df), 1)

    name_match = 0 if any(
        pattern.search(column) for pattern in TARGET_PATTERNS
    ) else 1

    # Classification targets normally have far fewer unique values
    # than observations.
    classification_like = (
        unique_count >= 2
        and unique_count <= min(50, max(2, int(row_count * 0.20)))
    )

    classification_penalty = 0 if classification_like else 1

    uniqueness_ratio = unique_count / row_count

    return (
        name_match,
        classification_penalty,
        uniqueness_ratio,
        column.lower(),
    )


def detect_target_candidates(df: pd.DataFrame) -> list[str]:
    """
    Return likely supervised-learning target columns.

    The function only suggests candidates. It does not silently choose a
    target unless the caller explicitly asks for automatic selection.
    """
    if df.empty:
        return []

    candidates: list[str] = []

    for raw_column in df.columns:
        column = str(raw_column)
        series = df[raw_column]

        non_null_unique = int(series.nunique(dropna=True))

        if non_null_unique < 2:
            continue

        name_match = any(
            pattern.search(column)
            for pattern in TARGET_PATTERNS
        )

        classification_like = (
            non_null_unique <= min(
                50,
                max(2, int(max(len(df), 1) * 0.20)),
            )
        )

        if name_match or classification_like:
            candidates.append(column)

    return sorted(
        candidates,
        key=lambda column: _target_score(df, column),
    )


def detect_id_candidates(
    df: pd.DataFrame,
    exclude: set[str] | None = None,
) -> list[str]:
    """
    Detect likely identifier / metadata columns.

    Conservative behavior:
      - name-based detection
      - exact monotonic integer index detection
      - exact 0..N-1 / 1..N sequences

    A generic unique numeric feature is not removed merely because it is
    unique.
    """
    if df.empty:
        return []

    excluded = {str(value) for value in (exclude or set())}
    candidates: list[str] = []

    for raw_column in df.columns:
        column = str(raw_column)

        if column in excluded:
            continue

        series = df[raw_column]

        name_match = bool(ID_PATTERNS.search(column))
        metadata_match = bool(METADATA_PATTERNS.search(column))
        index_match = _is_monotonic_integer_index(series)

        if name_match or metadata_match or index_match:
            candidates.append(column)

    return candidates


def detect_metadata_columns(
    df: pd.DataFrame,
    exclude: set[str] | None = None,
) -> list[str]:
    """
    Return likely metadata columns.

    This is separate from ID detection because metadata should be shown to
    the user as a reason for optional dropping rather than being treated as
    a predictive feature.
    """
    if df.empty:
        return []

    excluded = {str(value) for value in (exclude or set())}

    result = []

    for raw_column in df.columns:
        column = str(raw_column)

        if column in excluded:
            continue

        if METADATA_PATTERNS.search(column):
            result.append(column)

    return result


def _detect_datetime_columns(
    df: pd.DataFrame,
    excluded: set[str],
) -> list[str]:
    """
    Detect explicit datetime dtypes and strongly datetime-like object columns.
    """
    result: list[str] = []

    for raw_column in df.columns:
        column = str(raw_column)

        if column in excluded:
            continue

        series = df[raw_column]

        if pd.api.types.is_datetime64_any_dtype(series):
            result.append(column)
            continue

        if not pd.api.types.is_object_dtype(series):
            continue

        sample = series.dropna().head(200)

        if sample.empty:
            continue

        parsed = pd.to_datetime(
            sample,
            errors="coerce",
            utc=False,
        )

        if parsed.notna().mean() >= 0.90:
            result.append(column)

    return result

def detect_datetime_columns(
    df: pd.DataFrame,
    exclude: set[str] | None = None,
    excluded: set[str] | None = None,
) -> list[str]:
    """
    Public compatibility wrapper for datetime-column detection.

    Supports both `exclude=` and `excluded=` so the preprocessing
    layer can use this function without depending on the private
    implementation name.
    """
    excluded_columns: set[str] = set()

    if exclude:
        excluded_columns.update(
            str(column) for column in exclude
        )

    if excluded:
        excluded_columns.update(
            str(column) for column in excluded
        )

    return _detect_datetime_columns(
        df,
        excluded=excluded_columns,
    )


def _class_distribution(
    series: pd.Series,
) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in series.value_counts(
            dropna=False
        ).items()
    }


def profile_dataset(
    df: pd.DataFrame,
    target_column: str | None = None,
) -> DatasetProfile:
    """
    Analyze a tabular dataset for the Dataset page and preprocessing layer.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    clean = df.copy()
    clean.columns = [str(column) for column in clean.columns]

    target = (
        str(target_column)
        if target_column is not None
        else None
    )

    if target is not None and target not in clean.columns:
        raise ValueError(
            f"Target column not found: {target}"
        )

    all_columns = _safe_name_list(clean.columns)

    missing_values = {
        str(column): int(value)
        for column, value in clean.isna().sum().items()
        if int(value) > 0
    }

    numeric_columns = _safe_name_list(
        clean.select_dtypes(include=["number"]).columns
    )

    possible_id_columns = detect_id_candidates(
        clean,
        exclude={target} if target else None,
    )

    datetime_columns = _detect_datetime_columns(
        clean,
        excluded=set(possible_id_columns)
        | ({target} if target else set()),
    )

    datetime_set = set(datetime_columns)
    numeric_set = set(numeric_columns)

    categorical_columns = [
        column
        for column in all_columns
        if column not in numeric_set
        and column not in datetime_set
    ]

    constant_columns = [
        str(column)
        for column in clean.columns
        if int(clean[column].nunique(dropna=False)) <= 1
    ]

    possible_target_columns = detect_target_candidates(clean)

    class_distribution: dict[str, int] = {}

    if target is not None:
        class_distribution = _class_distribution(
            clean[target]
        )

    return DatasetProfile(
        rows=int(len(clean)),
        columns=int(clean.shape[1]),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
        constant_columns=constant_columns,
        possible_id_columns=possible_id_columns,
        possible_target_columns=possible_target_columns,
        missing_values=missing_values,
        duplicate_rows=int(clean.duplicated().sum()),
        class_distribution=class_distribution,
        target_column=target,
        dropped_columns=[],
        total_missing_cells=int(
            clean.isna().sum().sum()
        ),
        numeric_count=len(numeric_columns),
        categorical_count=len(categorical_columns),
        datetime_count=len(datetime_columns),
        constant_count=len(constant_columns),
        target_class_count=len(class_distribution),
    )