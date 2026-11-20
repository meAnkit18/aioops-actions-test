# MLOps Practical Assessment — Model Quality Monitoring

## Project
Iris multiclass classification (setosa, versicolor, virginica), using the repository's existing RandomForestClassifier champion model and FastAPI prediction service.

## Workflow
1. Generate a deterministic labelled incoming batch with unique record IDs.
2. Validate the schema, missing values, duplicate IDs, numeric features, and legal target labels; print row/column count and target distribution.
3. Load the existing champion model and evaluate it on the new labelled batch. Save accuracy, macro precision, macro recall, macro-F1, and a per-class classification report.
4. Combine the original Iris training dataset with the new labelled batch and train a separate Random Forest candidate using the same four input features.
5. Log candidate parameters, holdout metrics, and model artifact in MLflow.
6. Compare the champion and candidate on the same incoming labelled batch. The candidate must achieve accuracy >= 0.90 and macro-F1 no more than 0.01 below the champion to be eligible.
7. Save the candidate and a quality-gate JSON decision. The existing champion file is never overwritten automatically.

## Metric justification
Macro-F1 is the primary selection metric because this is a three-class classification task and all species should be treated equally. Accuracy alone can hide weaker performance on one class. The 0.90 accuracy floor is an additional guardrail, and the non-regression margin prevents replacing the champion with a meaningfully worse model.

## Execution evidence
Run:
```bash
python -m monitoring.generate_synthetic_data
python -m monitoring.run_monitoring
python -m pytest -v
```

Evidence is written to `artifacts/monitoring/`, the candidate model is `artifacts/monitoring/candidate_model.joblib`, and MLflow runs are stored in `mlflow_monitoring.db` (SQLite tracking backend). GitHub Actions also runs this workflow on dispatch or relevant pushes and uploads the evidence as a downloadable workflow artifact.
