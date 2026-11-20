import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from monitoring.run_monitoring import FEATURES, REQUIRED, validate_batch

def test_batch_validation_accepts_valid_rows(tmp_path):
    iris = load_iris(as_frame=True)
    frame = iris.data.copy()
    frame.columns = FEATURES
    frame["target"] = iris.target.astype(int)
    frame.insert(0, "record_id", [f"id-{i}" for i in range(len(frame))])
    path = tmp_path / "batch.csv"
    frame.to_csv(path, index=False)
    result = validate_batch(path)
    assert len(result) == 150
    assert list(result[REQUIRED].columns) == REQUIRED

def test_batch_validation_rejects_duplicate_ids(tmp_path):
    frame = pd.DataFrame({
        "record_id": ["same", "same"],
        "sepal_length": [5.1, 5.2],
        "sepal_width": [3.5, 3.4],
        "petal_length": [1.4, 1.5],
        "petal_width": [0.2, 0.2],
        "target": [0, 0],
    })
    path = tmp_path / "duplicate.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate record_id"):
        validate_batch(path)

def test_batch_validation_rejects_missing_values(tmp_path):
    frame = pd.DataFrame({
        "record_id": ["id-1"],
        "sepal_length": [None],
        "sepal_width": [3.5],
        "petal_length": [1.4],
        "petal_width": [0.2],
        "target": [0],
    })
    path = tmp_path / "missing.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing values"):
        validate_batch(path)
