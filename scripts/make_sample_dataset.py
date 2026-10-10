from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_sample(path: str = "data/sample/sample_kitsune.csv", rows: int = 600, features: int = 12) -> Path:
    rng = np.random.default_rng(42)
    n_attack = rows // 2
    n_normal = rows - n_attack
    X0 = rng.normal(0.0, 1.0, size=(n_normal, features))
    X1 = rng.normal(1.0, 1.0, size=(n_attack, features))
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_attack, dtype=int)])
    order = rng.permutation(rows)
    df = pd.DataFrame(X[order], columns=[f"F{i}" for i in range(1, features + 1)])
    df["Label"] = y[order]
    df["Scenario"] = "synthetic_smoke"
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print(f"Created sample dataset: {p} ({rows} rows, {features} features)")
    return p


if __name__ == "__main__":
    make_sample()
