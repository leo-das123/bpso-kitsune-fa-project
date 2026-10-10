from __future__ import annotations

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathlib import Path
import pandas as pd

from bpso_ids.config import make_config
from bpso_ids.pipeline import ResearchPipeline


if __name__ == "__main__":
    dataset = Path("data/sample/sample_kitsune.csv")
    if not dataset.exists():
        raise SystemExit("Run python scripts/make_sample_dataset.py first.")
    pipeline = ResearchPipeline(make_config("demo"))
    pipeline.load_and_prepare(dataset)
    pipeline.load_artifacts()
    df = pd.read_csv(dataset)
    record = df.drop(columns=["Label", "Scenario"]).iloc[[0]]
    print(pipeline.predict_record(record))
