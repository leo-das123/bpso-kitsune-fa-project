from __future__ import annotations

import csv
import math
import statistics
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ARCHIVE = (
    ROOT
    / "data"
    / "raw"
    / "kitsune"
    / "1.archive"
)

DATASET = "Mirai Botnet/Mirai_dataset.csv"
LABELS = "Mirai Botnet/mirai_labels.csv"

SAMPLE_ROWS = 10_000


def to_float(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main():

    print("=" * 100)
    print("KITSUNE 116-COLUMN DIAGNOSTIC")
    print("=" * 100)

    with zipfile.ZipFile(ARCHIVE, "r") as z:

        print(f"Dataset : {DATASET}")
        print(f"Labels  : {LABELS}")
        print()

        with (
            z.open(DATASET) as data_raw,
            z.open(LABELS) as label_raw,
        ):

            data_reader = csv.reader(
                line.decode("utf-8", errors="replace")
                for line in data_raw
            )

            label_reader = csv.reader(
                line.decode("utf-8", errors="replace")
                for line in label_raw
            )

            columns = []

            first = next(data_reader)

            column_count = len(first)

            print(f"Detected columns: {column_count}")
            print()

            for _ in range(column_count):
                columns.append([])

            labels = []

            for row_number in range(SAMPLE_ROWS):

                try:
                    row = first if row_number == 0 else next(data_reader)
                    label_row = next(label_reader)
                except StopIteration:
                    break

                if len(row) != column_count:
                    continue

                label_value = label_row[0].strip()

                if label_value == "":
                    continue

                labels.append(label_value)

                for i, value in enumerate(row):

                    number = to_float(value)

                    columns[i].append(number)

            print(
                f"Rows inspected: {len(labels):,}"
            )

        # ---------------------------------------------------------
        # Label distribution
        # ---------------------------------------------------------

        print()
        print("=" * 100)
        print("LABEL DISTRIBUTION")
        print("=" * 100)

        label_counts = Counter(labels)

        for label, count in label_counts.items():

            percentage = (
                count / len(labels) * 100
                if labels
                else 0
            )

            print(
                f"{label!r:>10}: "
                f"{count:>8,} "
                f"({percentage:6.2f}%)"
            )

        # ---------------------------------------------------------
        # Column statistics
        # ---------------------------------------------------------

        print()
        print("=" * 100)
        print("COLUMN STATISTICS")
        print("=" * 100)

        print(
            f"{'Column':>8} "
            f"{'Unique':>10} "
            f"{'Min':>20} "
            f"{'Max':>20} "
            f"{'Mean':>20} "
            f"{'Constant':>10}"
        )

        print("-" * 100)

        stats = []

        for i, values in enumerate(columns, start=1):

            cleaned = [
                x for x in values
                if x is not None
                and math.isfinite(x)
            ]

            unique = len(set(cleaned))

            if cleaned:

                minimum = min(cleaned)
                maximum = max(cleaned)
                mean = statistics.fmean(cleaned)

            else:

                minimum = float("nan")
                maximum = float("nan")
                mean = float("nan")

            constant = (
                len(set(cleaned)) <= 1
                if cleaned
                else True
            )

            stats.append(
                {
                    "column": i,
                    "unique": unique,
                    "min": minimum,
                    "max": maximum,
                    "mean": mean,
                    "constant": constant,
                }
            )

            print(
                f"{i:8d} "
                f"{unique:10d} "
                f"{minimum:20.6g} "
                f"{maximum:20.6g} "
                f"{mean:20.6g} "
                f"{str(constant):>10}"
            )

        # ---------------------------------------------------------
        # Test whether any feature column equals the label.
        # ---------------------------------------------------------

        print()
        print("=" * 100)
        print("LABEL-LEAKAGE CHECK")
        print("=" * 100)

        numeric_labels = []

        for value in labels:
            number = to_float(value)

            if number is None:
                numeric_labels.append(None)
            else:
                numeric_labels.append(number)

        for i, values in enumerate(columns, start=1):

            if len(values) != len(numeric_labels):
                continue

            exact_matches = 0
            inverse_matches = 0
            valid = 0

            for x, y in zip(values, numeric_labels):

                if x is None or y is None:
                    continue

                valid += 1

                if x == y:
                    exact_matches += 1

                if x == 1 - y:
                    inverse_matches += 1

            if valid == 0:
                continue

            exact_ratio = exact_matches / valid
            inverse_ratio = inverse_matches / valid

            if exact_ratio > 0.95 or inverse_ratio > 0.95:

                print(
                    f"Column {i}: "
                    f"possible label leakage "
                    f"(exact={exact_ratio:.4f}, "
                    f"inverse={inverse_ratio:.4f})"
                )

        # ---------------------------------------------------------
        # Time-like candidates
        # ---------------------------------------------------------

        print()
        print("=" * 100)
        print("POSSIBLE TIME-LIKE COLUMNS")
        print("=" * 100)

        for s in stats:

            if (
                s["min"] > 1_000_000_000
                and s["max"] < 2_000_000_000
            ):
                print(
                    f"Column {s['column']}: "
                    f"min={s['min']}, "
                    f"max={s['max']}, "
                    f"unique={s['unique']}"
                )

        # ---------------------------------------------------------
        # Constant columns
        # ---------------------------------------------------------

        print()
        print("=" * 100)
        print("CONSTANT COLUMNS")
        print("=" * 100)

        constant_columns = [
            s["column"]
            for s in stats
            if s["constant"]
        ]

        print(constant_columns)

        print()
        print("=" * 100)
        print("DIAGNOSTIC COMPLETE")
        print("=" * 100)


if __name__ == "__main__":
    main()