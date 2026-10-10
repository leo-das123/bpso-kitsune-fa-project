from __future__ import annotations

from pathlib import Path
import sys

import kagglehub


# Project root:
# D:\PYTHON\bpso_kitsune_fa_project\bpso_kitsune_fa_project
ROOT = Path(__file__).resolve().parents[1]

# Store the full Kitsune dataset inside the project on D:
DATASET_DIR = ROOT / "data" / "raw" / "kitsune"

DATASET = "ymirsky/network-attack-dataset-kitsune"


def main() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("KITSUNE DATASET DOWNLOAD")
    print("=" * 70)
    print(f"Project root : {ROOT}")
    print(f"Download to  : {DATASET_DIR}")
    print()

    path = kagglehub.dataset_download(
        DATASET,
        output_dir=str(DATASET_DIR),
    )

    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"Dataset path: {path}")
    print()
    print("Next step:")
    print("python scripts/inspect_dataset.py --root", path)


if __name__ == "__main__":
    main()