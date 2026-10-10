from __future__ import annotations

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


def main() -> None:

    if not ARCHIVE.exists():
        raise FileNotFoundError(
            f"Archive not found:\n{ARCHIVE}"
        )

    with zipfile.ZipFile(ARCHIVE, "r") as z:

        target = "Mirai Botnet/Mirai_dataset.csv"

        if target not in z.namelist():
            print("Available dataset files:")
            for name in z.namelist():
                if name.lower().endswith("_dataset.csv"):
                    print(name)
            raise FileNotFoundError(
                f"Could not find:\n{target}"
            )

        print("=" * 90)
        print("KITSUNE SCHEMA INSPECTION")
        print("=" * 90)
        print(f"File: {target}")
        print()

        with z.open(target) as raw:

            reader = csv.reader(
                line.decode(
                    "utf-8",
                    errors="replace"
                )
                for line in raw
            )

            first_row = next(reader)

            print("Column count:", len(first_row))
            print()

            print("FIRST 10 VALUES")
            print("-" * 90)

            for i, value in enumerate(first_row[:10]):
                print(f"{i + 1:>4}: {value}")

            print()
            print("LAST 10 VALUES")
            print("-" * 90)

            start = max(0, len(first_row) - 10)

            for i in range(start, len(first_row)):
                print(f"{i + 1:>4}: {first_row[i]}")

            print()
            print("COMPLETE FIRST ROW")
            print("-" * 90)

            print(first_row)


if __name__ == "__main__":
    main()