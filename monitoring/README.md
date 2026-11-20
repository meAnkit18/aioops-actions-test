# Model quality monitoring and retraining

This assessment extension is built for the repository's existing Iris multiclass classification API and Random Forest champion model.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m monitoring.generate_synthetic_data
python -m monitoring.run_monitoring
```

The synthetic incoming batch is deterministic, labelled, and generated from Iris observations with small measurement drift. It is for a reproducible classroom demonstration, not real production evidence.

## Outputs

- `data/incoming_iris_batch.csv`: 180 labelled incoming records with unique IDs.
- `data/iris_training_data.csv`: original labelled Iris training data snapshot.
- `artifacts/monitoring/evaluation_report.json`: validation, class distribution, champion metrics, candidate metrics, and MLflow run ID.
- `artifacts/monitoring/candidate_model.joblib`: separately saved retrained candidate.
- `artifacts/monitoring/quality_gate_decision.json`: explicit, auditable promotion decision.

MLflow runs are stored in the local SQLite database `mlflow_monitoring.db` and can be viewed with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow_monitoring.db --host 127.0.0.1 --port 6001
```

## Quality gate

Macro-F1 is the primary metric because it gives equal importance to each of the three Iris classes, rather than allowing the largest class to dominate. The candidate must also achieve at least 0.90 accuracy on the labelled incoming batch and may not reduce macro-F1 by more than 0.01 versus the current champion on that same batch.

The script **never overwrites** `models/irifulls_model.joblib`. A passing candidate is marked eligible for manual promotion; a failing candidate leaves the champion untouched. The JSON decision records the comparison, thresholds, action, and MLflow run ID.

## Assessment mapping

- A: required columns, missing values, duplicate IDs, batch size, and target distribution are checked/displayed.
- B: current champion is evaluated on incoming labelled data; metrics and classification report are saved.
- C: original labelled Iris data is combined with the incoming batch; candidate is trained and tracked in MLflow.
- D: macro-F1 + accuracy quality gate writes an auditable decision; champion is never automatically replaced.
- E: generated data, reports, model artifact, decision JSON, and MLflow run are reproducible via the commands above.
