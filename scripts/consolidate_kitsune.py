from __future__ import annotations

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from bpso_ids.data_io import consolidate_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default="data/processed/kitsune_consolidated.csv")
    args = parser.parse_args()

    output = consolidate_dataset(args.root, args.output)
    print(f"Saved consolidated dataset: {output}")
