from __future__ import annotations

from pathlib import Path
from typing import Iterable
import re

import numpy as np
import pandas as pd

from .schema import (
    DatasetProfile,
    detect_target_candidates,
    profile_dataset,
)


# ---------------------------------------------------------------------------
# Generic CSV loading
# ---------------------------------------------------------------------------

LABEL_RE = re.compile(
    r"(label|labels|target|class|attack|anomaly|"
    r"outcome|result|response|diagnosis|disease|status|y)$",
    re.IGNORECASE,
)

KITSUNE_DATASET_RE = re.compile(
    r"_dataset$",
    re.IGNORECASE,
)

KITSUNE_LABEL_RE = re.compile(
    r"_labels?$",
    re.IGNORECASE,
)


def list_csv_files(
    root: str | Path,
) -> list[Path]:
    """
    Recursively list CSV files under a directory.
    """
    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {root}"
        )

    if not root.is_dir():
        raise NotADirectoryError(
            f"Expected a directory: {root}"
        )

    return sorted(
        path
        for path in root.rglob("*.csv")
        if path.is_file()
    )


def load_csv(
    path: str | Path,
    **kwargs: object,
) -> pd.DataFrame:
    """
    Generic CSV loader.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"CSV file does not exist: {path}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Expected a CSV file, got: {path.name}"
        )

    return pd.read_csv(path, **kwargs)


def load_tabular_dataset(
    path: str | Path,
    *,
    target_column: str | None = None,
    **read_csv_kwargs: object,
) -> tuple[pd.DataFrame, DatasetProfile]:
    """
    Load one generic user CSV and immediately profile it.
    """
    df = load_csv(
        path,
        **read_csv_kwargs,
    )

    profile = profile_dataset(
        df,
        target_column=target_column,
    )

    return df, profile


def inspect_csv_files(
    root: str | Path,
) -> pd.DataFrame:
    """
    Create a lightweight directory-level CSV inventory.
    """
    rows: list[dict[str, object]] = []

    for path in list_csv_files(root):
        try:
            preview = pd.read_csv(
                path,
                nrows=1000,
            )

            profile = profile_dataset(
                preview
            )

            rows.append(
                {
                    "file": str(path),
                    "rows_preview": int(len(preview)),
                    "columns": int(preview.shape[1]),
                    "numeric_columns": len(
                        profile.numeric_columns
                    ),
                    "categorical_columns": len(
                        profile.categorical_columns
                    ),
                    "missing_cells_preview": int(
                        preview.isna().sum().sum()
                    ),
                    "possible_targets": ", ".join(
                        profile.possible_target_columns
                    ),
                    "possible_ids": ", ".join(
                        profile.possible_id_columns
                    ),
                    "size_mb": round(
                        path.stat().st_size
                        / (1024 * 1024),
                        3,
                    ),
                }
            )

        except Exception as exc:
            rows.append(
                {
                    "file": str(path),
                    "error": str(exc),
                }
            )

    return pd.DataFrame(rows)


def infer_target_column(
    df: pd.DataFrame,
    explicit_target: str | None = None,
) -> str:
    """
    Resolve a target column.

    Explicit selection always wins.
    Otherwise use the highest-ranked schema candidate.
    """
    if explicit_target is not None:
        if explicit_target not in df.columns:
            raise ValueError(
                f"Target column not found: "
                f"{explicit_target}"
            )

        return str(explicit_target)

    candidates = detect_target_candidates(df)

    if not candidates:
        raise ValueError(
            "No target column could be inferred. "
            "Select the target column explicitly."
        )

    return str(candidates[0])


# ---------------------------------------------------------------------------
# Kitsune-specific loading
# ---------------------------------------------------------------------------

def _norm_stem(path: Path) -> str:
    """
    Normalize a Kitsune feature/label filename so that e.g.

        Mirai_dataset.csv
        mirai_labels.csv

    become comparable.
    """
    stem = path.stem.lower()

    stem = re.sub(
        r"[_\-\s]+(dataset|data|features?|labels?|label|y)$",
        "",
        stem,
    )

    return stem.strip()


def detect_kitsune_pairs(
    root: str | Path,
) -> list[tuple[Path, Path]]:
    """
    Detect Kitsune *_dataset.csv + *_labels.csv pairs.
    """
    csvs = list_csv_files(root)

    label_files = [
        path
        for path in csvs
        if KITSUNE_LABEL_RE.search(path.stem)
    ]

    feature_files = [
        path
        for path in csvs
        if KITSUNE_DATASET_RE.search(path.stem)
    ]

    pairs: list[tuple[Path, Path]] = []

    for feature_file in feature_files:
        exact_parent = [
            label
            for label in label_files
            if label.parent == feature_file.parent
            and _norm_stem(label)
            == _norm_stem(feature_file)
        ]

        candidates = exact_parent

        if not candidates:
            same_stem = [
                label
                for label in label_files
                if _norm_stem(label)
                == _norm_stem(feature_file)
            ]

            candidates = same_stem

        if not candidates:
            same_parent = [
                label
                for label in label_files
                if label.parent == feature_file.parent
            ]

            if len(same_parent) == 1:
                candidates = same_parent

        if candidates:
            pairs.append(
                (
                    feature_file,
                    candidates[0],
                )
            )

    return sorted(pairs)


# Backward-compatible alias.
def detect_pairs(
    root: str | Path,
) -> list[tuple[Path, Path]]:
    return detect_kitsune_pairs(root)


def _looks_like_index(
    series: pd.Series,
) -> bool:
    """
    Detect exact 0..N-1 or 1..N numeric index columns.
    """
    if series.empty:
        return False

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.notna().mean() < 0.999:
        return False

    values = numeric.to_numpy()

    zero_based = np.arange(
        len(values),
        dtype=float,
    )

    one_based = np.arange(
        1,
        len(values) + 1,
        dtype=float,
    )

    return bool(
        np.array_equal(values, zero_based)
        or np.array_equal(values, one_based)
    )


def read_feature_csv(
    path: str | Path,
) -> pd.DataFrame:
    """
    Read a headerless Kitsune feature CSV.

    The Mirai file in the downloaded Kaggle package contains one additional
    leading row-index column. It is removed only when it is demonstrably
    an exact monotonic row index.

    Other 115-column Kitsune files remain untouched.
    """
    path = Path(path)

    df = pd.read_csv(
        path,
        header=None,
    )

    if df.empty:
        raise ValueError(
            f"Feature CSV is empty: {path}"
        )

    if (
        df.shape[1] == 116
        and _looks_like_index(df.iloc[:, 0])
    ):
        df = df.iloc[:, 1:].copy()

    numeric = df.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if numeric.isna().all(axis=None):
        raise ValueError(
            f"No numeric feature data detected in: {path}"
        )

    numeric.columns = [
        f"F{i}"
        for i in range(
            1,
            numeric.shape[1] + 1,
        )
    ]

    return numeric


def read_label_csv(
    path: str | Path,
) -> pd.Series:
    """
    Read the first column of a headerless Kitsune label CSV.
    """
    path = Path(path)

    df = pd.read_csv(
        path,
        header=None,
    )

    if df.shape[1] == 0:
        raise ValueError(
            f"Empty label file: {path}"
        )

    return df.iloc[:, 0].rename("Label")


def consolidate_kitsune_pairs(
    pairs: Iterable[tuple[Path, Path]],
) -> pd.DataFrame:
    """
    Combine multiple Kitsune feature/label pairs into one dataframe.
    """
    blocks: list[pd.DataFrame] = []

    for feature_file, label_file in pairs:
        X = read_feature_csv(
            feature_file
        )

        y = read_label_csv(
            label_file
        )

        if len(X) != len(y):
            raise ValueError(
                f"Row mismatch for "
                f"{feature_file.name}: "
                f"features={len(X)}, "
                f"labels={len(y)}"
            )

        block = X.copy()

        block["Label"] = y.to_numpy()

        block["Scenario"] = (
            feature_file.parent.name
        )

        blocks.append(block)

    if not blocks:
        raise ValueError(
            "No Kitsune feature/label pairs were found."
        )

    return pd.concat(
        blocks,
        ignore_index=True,
    )


# Backward-compatible alias.
def consolidate_pairs(
    pairs: Iterable[tuple[Path, Path]],
) -> pd.DataFrame:
    return consolidate_kitsune_pairs(pairs)


def consolidate_kitsune(
    root: str | Path,
    output: str | Path,
) -> Path:
    """
    Consolidate Kitsune CSV pairs into a single CSV.
    """
    pairs = detect_kitsune_pairs(root)

    if not pairs:
        raise ValueError(
            "No Kitsune feature/label CSV pairs detected."
        )

    dataset = consolidate_kitsune_pairs(
        pairs
    )

    output = Path(output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_csv(
        output,
        index=False,
    )

    return output


# Backward-compatible alias.
def consolidate_dataset(
    root: str | Path,
    output: str | Path,
) -> Path:
    return consolidate_kitsune(
        root,
        output,
    )