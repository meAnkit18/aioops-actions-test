"""Validate, evaluate, retrain, and quality-gate an Iris classifier."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts" / "monitoring"
MODEL_DIR = ROOT / "models"
CHAMPION_PATH = MODEL_DIR / "irifulls_model.joblib"
CANDIDATE_PATH = ARTIFACT_DIR / "candidate_model.joblib"
BATCH_PATH = DATA_DIR / "incoming_iris_batch.csv"
TRAINING_DATA_PATH = DATA_DIR / "iris_training_data.csv"
REPORT_PATH = ARTIFACT_DIR / "evaluation_report.json"
DECISION_PATH = ARTIFACT_DIR / "quality_gate_decision.json"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "target"
REQUIRED = ["record_id", *FEATURES, TARGET]
MIN_ACCURACY = 0.90
MAX_ACCURACY_DROP = 0.01

def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

def validate_batch(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Incoming batch not found: {path}. Run python -m monitoring.generate_synthetic_data first.")
    df = pd.read_csv(path)
    missing_columns = sorted(set(REQUIRED) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if df[REQUIRED].isna().any().any():
        bad = df[REQUIRED].columns[df[REQUIRED].isna().any()].tolist()
        raise ValueError(f"Missing values detected in columns: {bad}")
    if df["record_id"].duplicated().any():
        duplicates = df.loc[df["record_id"].duplicated(), "record_id"].tolist()
        raise ValueError(f"Duplicate record_id values detected: {duplicates[:5]}")
    if df[FEATURES].apply(lambda col: pd.to_numeric(col, errors="coerce")).isna().any().any():
        raise ValueError("Feature columns must contain numeric values.")
    if not set(df[TARGET].unique()).issubset({0, 1, 2}):
        raise ValueError("Target must contain only Iris class labels 0, 1, or 2.")
    df[FEATURES] = df[FEATURES].astype(float)
    df[TARGET] = df[TARGET].astype(int)
    print(f"Incoming dataset: {len(df)} rows x {len(df.columns)} columns")
    print(f"Duplicate IDs: {int(df['record_id'].duplicated().sum())}")
    print("Target distribution:")
    print(df[TARGET].value_counts().sort_index().to_string())
    return df

def load_original_training_data() -> pd.DataFrame:
    if TRAINING_DATA_PATH.exists():
        original = pd.read_csv(TRAINING_DATA_PATH)
    else:
        iris = load_iris(as_frame=True)
        original = iris.data.copy()
        original.columns = FEATURES
        original[TARGET] = iris.target.astype(int)
        original.insert(0, "record_id", [f"original-{i:04d}" for i in range(1, len(original) + 1)])
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        original.to_csv(TRAINING_DATA_PATH, index=False)
    return original[REQUIRED].copy()

def main() -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    incoming = validate_batch(BATCH_PATH)

    if not CHAMPION_PATH.exists():
        raise FileNotFoundError(f"Existing champion model missing: {CHAMPION_PATH}")
    champion = joblib.load(CHAMPION_PATH)
    X_new, y_new = incoming[FEATURES], incoming[TARGET]
    champion_predictions = champion.predict(X_new.to_numpy())
    champion_metrics = metrics(y_new, champion_predictions)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(BATCH_PATH.relative_to(ROOT)),
        "rows": int(len(incoming)),
        "columns": list(incoming.columns),
        "target_distribution": {str(k): int(v) for k, v in incoming[TARGET].value_counts().sort_index().items()},
        "validation": {"required_columns_present": True, "missing_values": 0, "duplicate_record_ids": 0},
        "champion_metrics": champion_metrics,
        "champion_classification_report": classification_report(y_new, champion_predictions, output_dict=True, zero_division=0),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Retrain on original labelled training set + incoming labelled batch.
    combined = pd.concat([load_original_training_data(), incoming[REQUIRED]], ignore_index=True)
    combined = combined.drop_duplicates(subset="record_id", keep="last")
    X, y = combined[FEATURES], combined[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    candidate = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
    # SQLite is the supported MLflow tracking backend; avoid the deprecated file-store backend.
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow_monitoring.db'}")
    mlflow.set_experiment("iris-model-quality-monitoring")
    with mlflow.start_run(run_name="iris-retraining-candidate") as run:
        mlflow.log_params({
            "model_type": "RandomForestClassifier",
            "n_estimators": 150,
            "random_state": 42,
            "training_rows": int(len(X_train)),
            "combined_rows": int(len(combined)),
            "incoming_rows": int(len(incoming)),
        })
        candidate.fit(X_train, y_train)
        validation_metrics = metrics(y_test, candidate.predict(X_test))
        mlflow.log_metrics({f"candidate_{k}": v for k, v in validation_metrics.items()})
        # This candidate is trained in this workflow from the trusted Iris dataset.
        # skops flags scikit-learn's internal tree storage type; trust only that exact type.
        mlflow.sklearn.log_model(
            candidate,
            name="candidate-model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        candidate_run_id = run.info.run_id

    joblib.dump(candidate, CANDIDATE_PATH)
    candidate_new_metrics = metrics(y_new, candidate.predict(X_new))
    report["candidate_holdout_metrics"] = validation_metrics
    report["candidate_incoming_batch_metrics"] = candidate_new_metrics
    report["candidate_mlflow_run_id"] = candidate_run_id
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Quality gate uses macro-F1 as the primary criterion because all three
    # Iris classes matter; accuracy is an additional minimum safety floor.
    champion_f1 = champion_metrics["f1_macro"]
    candidate_f1 = candidate_new_metrics["f1_macro"]
    accuracy_ok = candidate_new_metrics["accuracy"] >= MIN_ACCURACY
    no_regression = candidate_f1 >= champion_f1 - MAX_ACCURACY_DROP
    eligible = bool(accuracy_ok and no_regression)
    decision = {
        "decided_at_utc": datetime.now(timezone.utc).isoformat(),
        "champion_model": str(CHAMPION_PATH.relative_to(ROOT)),
        "candidate_model": str(CANDIDATE_PATH.relative_to(ROOT)),
        "primary_metric": "f1_macro",
        "reason": "Macro-F1 weights all three Iris classes equally; accuracy is used as a minimum guardrail.",
        "criteria": {
            "candidate_accuracy_minimum": MIN_ACCURACY,
            "candidate_accuracy": candidate_new_metrics["accuracy"],
            "candidate_f1_macro": candidate_f1,
            "champion_f1_macro": champion_f1,
            "maximum_allowed_f1_drop": MAX_ACCURACY_DROP,
            "accuracy_passed": bool(accuracy_ok),
            "f1_non_regression_passed": bool(no_regression),
        },
        "candidate_eligible_for_promotion": eligible,
        "action": "ELIGIBLE_FOR_MANUAL_PROMOTION" if eligible else "RETAIN_EXISTING_CHAMPION",
        "champion_overwritten": False,
        "candidate_mlflow_run_id": candidate_run_id,
    }
    DECISION_PATH.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print("\nChampion on incoming batch:", json.dumps(champion_metrics, indent=2))
    print("Candidate on incoming batch:", json.dumps(candidate_new_metrics, indent=2))
    print(f"Candidate holdout metrics: {json.dumps(validation_metrics)}")
    print(f"QUALITY GATE: {decision['action']}")
    print(f"Report: {REPORT_PATH}")
    print(f"Candidate: {CANDIDATE_PATH}")
    print(f"Decision: {DECISION_PATH}")
    return decision

if __name__ == "__main__":
    main()
