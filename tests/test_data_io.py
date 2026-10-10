from pathlib import Path
import pandas as pd

from bpso_ids.data_io import detect_pairs, consolidate_pairs


def test_detect_pairs_and_consolidate(tmp_path: Path):
    folder = tmp_path / "scenario"
    folder.mkdir()
    X = pd.DataFrame([[1, 2], [3, 4]])
    y = pd.DataFrame([0, 1])
    X.to_csv(folder / "scenario_dataset.csv", header=False, index=False)
    y.to_csv(folder / "scenario_labels.csv", header=False, index=False)

    pairs = detect_pairs(tmp_path)
    assert len(pairs) == 1
    df = consolidate_pairs(pairs)
    assert list(df.columns) == ["F1", "F2", "Label", "Scenario"]
    assert df["Label"].tolist() == [0, 1]
