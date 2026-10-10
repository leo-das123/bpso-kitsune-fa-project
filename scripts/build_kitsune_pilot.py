from __future__ import annotations

import argparse
import csv
import random
import zipfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ARCHIVE = (
    ROOT
    / "data"
    / "raw"
    / "kitsune"
    / "1.archive"
)

OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "kitsune_pilot.csv"
)


def is_header(row: list[str]) -> bool:
    """Detect an optional CSV header row."""
    if not row:
        return False

    first = row[0].strip().lower()

    return first in {
        "label",
        "labels",
        "class",
        "target",
        "feature",
        "f1",
    }


def normalize_label(value: str) -> str:
    return value.strip()


def find_dataset_label_pairs(
    archive: zipfile.ZipFile,
):
    """
    Find one *_dataset.csv and one *_labels.csv
    inside each scenario directory.
    """
    files = archive.namelist()

    dataset_files = [
        name
        for name in files
        if name.lower().endswith("_dataset.csv")
    ]

    label_files = [
        name
        for name in files
        if name.lower().endswith("_labels.csv")
    ]

    labels_by_parent = defaultdict(list)

    for name in label_files:
        parent = str(Path(name).parent)
        labels_by_parent[parent].append(name)

    pairs = []

    for dataset_name in dataset_files:
        parent = str(Path(dataset_name).parent)

        candidates = labels_by_parent.get(parent, [])

        if not candidates:
            print(
                f"WARNING: No label file found for:\n"
                f"  {dataset_name}"
            )
            continue

        if len(candidates) > 1:
            print(
                f"WARNING: Multiple label files found for:\n"
                f"  {dataset_name}"
            )

        label_name = candidates[0]

        scenario = Path(dataset_name).parent.name

        pairs.append(
            (
                scenario,
                dataset_name,
                label_name,
            )
        )

    return sorted(pairs)


def reservoir_add(
    reservoir: list[list[str]],
    row: list[str],
    seen: int,
    capacity: int,
    rng: random.Random,
):
    """
    Reservoir sampling.

    Keeps only 'capacity' rows in memory while scanning
    a potentially very large CSV.
    """
    if len(reservoir) < capacity:
        reservoir.append(row)
        return

    index = rng.randint(0, seen - 1)

    if index < capacity:
        reservoir[index] = row


def sample_scenario(
    archive: zipfile.ZipFile,
    scenario: str,
    dataset_name: str,
    label_name: str,
    per_class: int,
    rng: random.Random,
):
    print()
    print("=" * 80)
    print(f"SCENARIO: {scenario}")
    print("=" * 80)

    reservoirs: dict[str, list[list[str]]] = defaultdict(list)
    seen: dict[str, int] = defaultdict(int)

    total_rows = 0
    feature_count = None

    with (
        archive.open(dataset_name) as dataset_raw,
        archive.open(label_name) as label_raw,
    ):
        dataset_stream = (
            line.decode("utf-8", errors="replace")
            for line in dataset_raw
        )

        label_stream = (
            line.decode("utf-8", errors="replace")
            for line in label_raw
        )

        dataset_reader = csv.reader(dataset_stream)
        label_reader = csv.reader(label_stream)

        first_dataset = next(dataset_reader, None)
        first_label = next(label_reader, None)

        if first_dataset is None or first_label is None:
            raise RuntimeError(
                f"Empty dataset/label file for {scenario}"
            )

        # Handle optional headers.
        if not is_header(first_dataset):
            dataset_first_row = first_dataset
        else:
            dataset_first_row = None

        if not is_header(first_label):
            label_first_row = first_label
        else:
            label_first_row = None

        pending_dataset = dataset_first_row
        pending_label = label_first_row

        while True:

            if pending_dataset is None:
                try:
                    feature_row = next(dataset_reader)
                except StopIteration:
                    break
            else:
                feature_row = pending_dataset
                pending_dataset = None

            if pending_label is None:
                try:
                    label_row = next(label_reader)
                except StopIteration:
                    raise RuntimeError(
                        f"Dataset contains more rows than labels "
                        f"for scenario: {scenario}"
                    )
            else:
                label_row = pending_label
                pending_label = None

            if not feature_row:
                continue

            if not label_row:
                continue

            label = normalize_label(label_row[0])

            if not label:
                continue

            if feature_count is None:
                feature_count = len(feature_row)

                print(
                    f"Detected feature count: {feature_count}"
                )

                if feature_count != 115:
                    raise ValueError(
                        f"Expected 115 features, "
                        f"but {scenario} contains "
                        f"{feature_count}."
                    )

            if len(feature_row) != feature_count:
                print(
                    f"WARNING: Invalid row length "
                    f"in {scenario}: "
                    f"{len(feature_row)}"
                )
                continue

            total_rows += 1

            seen[label] += 1

            reservoir_add(
                reservoirs[label],
                feature_row,
                seen[label],
                per_class,
                rng,
            )

            if total_rows % 100_000 == 0:
                print(
                    f"Scanned {total_rows:,} rows..."
                )

    print()
    print(f"Total rows scanned: {total_rows:,}")

    print("Available labels:")

    for label, count in sorted(seen.items()):
        print(
            f"  {label!r}: "
            f"{count:,} rows"
        )

    if len(reservoirs) != 2:
        raise ValueError(
            f"{scenario} has {len(reservoirs)} unique labels. "
            f"Our pilot builder expects a binary classification "
            f"scenario."
        )

    result_rows = []

    for label, rows in sorted(reservoirs.items()):

        if len(rows) < per_class:
            print(
                f"WARNING: Requested {per_class} "
                f"rows for label {label!r}, "
                f"but only {len(rows)} are available."
            )

        for row in rows:
            result_rows.append(
                row + [label, scenario]
            )

    rng.shuffle(result_rows)

    print(
        f"Selected pilot rows: {len(result_rows):,}"
    )

    return result_rows, feature_count


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a balanced Kitsune pilot dataset "
            "directly from the compressed archive."
        )
    )

    parser.add_argument(
        "--per-class",
        type=int,
        default=2500,
        help=(
            "Rows per class per scenario. "
            "Default: 2500"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )

    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help=(
            "Scenario names to process. "
            "If omitted, all scenarios are processed."
        ),
    )

    args = parser.parse_args()

    if not ARCHIVE.exists():
        raise FileNotFoundError(
            f"Kitsune archive not found:\n{ARCHIVE}"
        )

    if args.per_class <= 0:
        raise ValueError(
            "--per-class must be greater than zero."
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = random.Random(args.seed)

    print("=" * 80)
    print("KITSUNE PILOT DATASET BUILDER")
    print("=" * 80)

    print(f"Archive       : {ARCHIVE}")
    print(f"Output        : {OUTPUT}")
    print(f"Rows/class    : {args.per_class}")
    print(f"Random seed   : {args.seed}")

    with zipfile.ZipFile(
        ARCHIVE,
        "r",
    ) as archive:

        pairs = find_dataset_label_pairs(
            archive
        )

        print()
        print(
            f"Scenarios found in archive: {len(pairs)}"
        )

        if args.scenarios:
            requested = {
                name.strip().lower()
                for name in args.scenarios
            }

            pairs = [
                pair
                for pair in pairs
                if pair[0].strip().lower()
                in requested
            ]

        if not pairs:
            raise RuntimeError(
                "No matching scenarios found."
            )

        print("Scenarios selected:")

        for scenario, _, _ in pairs:
            print(f"  - {scenario}")

        all_rows = []
        feature_count = None

        for (
            scenario,
            dataset_name,
            label_name,
        ) in pairs:

            rows, current_feature_count = (
                sample_scenario(
                    archive=archive,
                    scenario=scenario,
                    dataset_name=dataset_name,
                    label_name=label_name,
                    per_class=args.per_class,
                    rng=rng,
                )
            )

            if feature_count is None:
                feature_count = current_feature_count

            elif feature_count != current_feature_count:
                raise ValueError(
                    "Feature count differs between scenarios."
                )

            all_rows.extend(rows)

    rng.shuffle(all_rows)

    header = [
        f"F{i}"
        for i in range(1, feature_count + 1)
    ]

    header += [
        "Label",
        "Scenario",
    ]

    print()
    print("=" * 80)
    print("WRITING PILOT DATASET")
    print("=" * 80)

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(header)
        writer.writerows(all_rows)

    print()
    print(f"Saved: {OUTPUT}")
    print(f"Rows : {len(all_rows):,}")
    print(f"Cols : {len(header)}")
    print()
    print("Dataset structure:")
    print("  115 feature columns")
    print("  1 label column")
    print("  1 scenario column")


if __name__ == "__main__":
    main()