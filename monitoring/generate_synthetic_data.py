"""Generate a deterministic, labelled incoming Iris batch for the assessment."""
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.utils import check_random_state

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT = DATA_DIR / "incoming_iris_batch.csv"

def generate_batch(n_samples: int = 180, random_state: int = 2026) -> Path:
    iris = load_iris(as_frame=True)
    X = iris.data.copy()
    y = iris.target.copy()
    rng = check_random_state(random_state)

    # Sample real Iris feature/label pairs, then apply small deterministic
    # measurement drift to emulate a later incoming data batch.
    indices = rng.randint(0, len(X), size=n_samples)
    batch = X.iloc[indices].reset_index(drop=True)
    labels = y.iloc[indices].reset_index(drop=True)
    drift = [0.10, -0.06, 0.12, 0.08]
    for column, offset in zip(batch.columns, drift):
        batch[column] = (batch[column] + offset + rng.normal(0, 0.035, n_samples)).clip(lower=0.05)
    batch.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    batch["target"] = labels.astype(int)
    batch.insert(0, "record_id", [f"incoming-{i:04d}" for i in range(1, n_samples + 1)])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    batch.to_csv(OUTPUT, index=False)
    print(f"Generated {len(batch)} labelled records: {OUTPUT}")
    print("Target distribution:")
    print(batch["target"].value_counts().sort_index().to_string())
    return OUTPUT

if __name__ == "__main__":
    generate_batch()
