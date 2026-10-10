import numpy as np
import pandas as pd

from bpso_ids.preprocessing import Preprocessor


def test_preprocessor_handles_nan_and_split():
    df = pd.DataFrame({
        "F1": [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10],
        "F2": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "Label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "Scenario": ["s"] * 10,
    })
    prep = Preprocessor(42)
    prepared = prep.prepare(df)
    assert prepared.X_train.shape[1] == 2
    assert np.isfinite(prepared.X_train).all()
    assert np.isfinite(prepared.X_test).all()
