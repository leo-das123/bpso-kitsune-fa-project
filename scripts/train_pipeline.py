from __future__ import annotations

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from pathlib import Path

from bpso_ids.config import make_config
from bpso_ids.pipeline import train_from_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mode", choices=["demo", "research"], default="demo")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--particles", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    args = parser.parse_args()

    cfg = make_config(args.mode)
    if args.max_rows is not None:
        cfg.max_rows = args.max_rows
    if args.particles is not None:
        cfg.n_particles = args.particles
    if args.iterations is not None:
        cfg.n_iterations = args.iterations

    print(f"Training mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    results = train_from_csv(args.dataset, cfg)
    print(json.dumps(results, indent=2))
    print("Artifacts saved under artifacts/")
