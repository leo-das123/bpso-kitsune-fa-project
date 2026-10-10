from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]

ARCHIVE = (
    ROOT
    / "data"
    / "raw"
    / "kitsune"
    / "1.archive"
)


def format_size(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]

    value = float(size)

    for unit in units:
        if value < 1024:
            return f"{value:.2f} {unit}"
        value /= 1024

    return f"{value:.2f} PB"


def main() -> None:
    if not ARCHIVE.exists():
        raise FileNotFoundError(
            f"Archive not found:\n{ARCHIVE}"
        )

    print("=" * 80)
    print("KITSUNE ARCHIVE INSPECTION")
    print("=" * 80)
    print(f"Archive: {ARCHIVE}")
    print()

    with zipfile.ZipFile(ARCHIVE, "r") as z:

        infos = z.infolist()

        csv_files = [
            info
            for info in infos
            if info.filename.lower().endswith(".csv")
        ]

        print(f"Total archive entries : {len(infos)}")
        print(f"CSV files             : {len(csv_files)}")
        print()

        total_csv_size = sum(
            info.file_size
            for info in csv_files
        )

        print(
            "Total uncompressed CSV size:",
            format_size(total_csv_size)
        )

        print()
        print("-" * 80)
        print("CSV FILES")
        print("-" * 80)

        for info in csv_files:
            print(
                f"{format_size(info.file_size):>12}  "
                f"{info.filename}"
            )


if __name__ == "__main__":
    main()