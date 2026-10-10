from __future__ import annotations

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from bpso_ids.data_io import inspect_csv_files, detect_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    print("CSV inspection")
    report = inspect_csv_files(args.root)
    if report.empty:
        print("No CSV files found.")
    else:
        print(report.to_string(index=False))

    print("\nDetected feature/label pairs:")
    for feature, label in detect_pairs(args.root):
        print(f"FEATURE: {feature}\nLABEL:   {label}\n")
