from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv
import zipfile


ROOT = Path(__file__).resolve().parents[1]

ARCHIVE = (
    ROOT
    / "data"
    / "raw"
    / "kitsune"
    / "1.archive"
)


def normalize_label(value: str) -> str:
    return value.strip()


def main() -> None:
    if not ARCHIVE.exists():
        raise FileNotFoundError(
            f"Kitsune archive not found:\n{ARCHIVE}"
        )

    print("=" * 90)
    print("KITSUNE LABEL INSPECTION")
    print("=" * 90)
    print(f"Archive: {ARCHIVE}")
    print()

    with zipfile.ZipFile(ARCHIVE, "r") as z:

        members = z.namelist()

        label_files = [
            name
            for name in members
            if name.lower().endswith("_labels.csv")
        ]

        dataset_files = [
            name
            for name in members
            if name.lower().endswith("_dataset.csv")
        ]

        print(f"Label files   : {len(label_files)}")
        print(f"Dataset files : {len(dataset_files)}")
        print()

        for label_name in sorted(label_files):

            print("-" * 90)
            print(f"Scenario: {Path(label_name).parent.name}")
            print(f"Label file: {label_name}")

            counts = Counter()
            row_count = 0

            with z.open(label_name) as raw:
                reader = csv.reader(
                    line.decode("utf-8", errors="replace")
                    for line in raw
                )

                for row in reader:
                    if not row:
                        continue

                    value = normalize_label(row[0])

                    # Skip a possible textual header.
                    if row_count == 0 and value.lower() in {
                        "label",
                        "labels",
                        "class",
                        "target",
                    }:
                        continue

                    counts[value] += 1
                    row_count += 1

            print(f"Rows: {row_count:,}")

            print("Label distribution:")
            for label, count in counts.items():
                percentage = (
                    count / row_count * 100
                    if row_count
                    else 0
                )

                print(
                    f"  {label!r:>12} : "
                    f"{count:>12,} "
                    f"({percentage:6.2f}%)"
                )

        print()
        print("=" * 90)
        print("FEATURE COLUMN CHECK")
        print("=" * 90)

        for dataset_name in sorted(dataset_files):

            with z.open(dataset_name) as raw:

                reader = csv.reader(
                    line.decode("utf-8", errors="replace")
                    for line in raw
                )

                first_row = next(reader, None)

            if first_row is None:
                print(f"{dataset_name}: EMPTY")
                continue

            print(
                f"{dataset_name}: "
                f"{len(first_row)} columns"
            )


if __name__ == "__main__":
    main()