# Iris Predictive Maintenance API — MLOps Monitoring & Retraining

A small end-to-end MLOps demonstration built around a **Random Forest classifier** for the Iris dataset. The repository contains a FastAPI prediction service and an automated model-quality workflow that validates new labelled data, evaluates the current model, trains a candidate, tracks the run with MLflow, and records a promotion decision.

> **Assessment focus:** model-quality monitoring, reproducible retraining, experiment tracking, and a safe champion/candidate workflow. The incoming batch is synthetic and intended for a classroom demonstration.

## What this project does

1. **Accepts an incoming labelled batch** and checks required columns, missing values, duplicate record IDs, numeric features, and valid target labels.
2. **Evaluates the current champion model** on the incoming batch using accuracy, macro precision, macro recall, and macro F1.
3. **Retrains a candidate model** using the original Iris training data combined with the incoming labelled batch.
4. **Tracks the candidate run in MLflow**, including parameters, validation metrics, and the serialized model.
5. **Applies a quality gate** using macro-F1 as the primary metric and an accuracy floor as a guardrail.
6. **Keeps the champion safe:** the workflow saves the candidate separately and never automatically overwrites the deployed champion model.

## Repository structure

```text
.
├── .github/workflows/
│   ├── ci.yaml                       # Existing CI checks
│   └── model-monitoring.yml          # Monitoring and retraining workflow
├── artifacts/monitoring/             # Generated reports and candidate artifact
├── data/                             # Original and generated labelled data
├── models/
│   └── irifulls_model.joblib         # Current champion model
├── monitoring/
│   ├── generate_synthetic_data.py    # Creates reproducible incoming data
│   ├── run_monitoring.py             # Validation, evaluation, retraining, gate
│   └── README.md                     # Detailed monitoring documentation
├── tests/
│   └── test_monitoring.py            # Data validation tests
├── main.py                           # FastAPI prediction service
├── train_model.py                    # Original model training script
└── requirements.txt
```

## Run the monitoring workflow locally

Use Python 3.10+ (the GitHub Actions workflow currently uses Python 3.12).

```bash
# 1. Clone the repository
git clone https://github.com/meAnkit18/aioops-actions-test.git
cd aioops-actions-test

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv .venv

# Windows PowerShell:
.venv\\Scripts\\Activate.ps1

# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Generate a deterministic labelled incoming batch
python -m monitoring.generate_synthetic_data

# 5. Run validation, evaluation, retraining, MLflow logging, and quality gate
python -m monitoring.run_monitoring

# 6. Run tests
python -m pytest -v
```

The data generator creates a reproducible batch of 180 labelled Iris records with unique IDs. It introduces small feature variation to simulate incoming data. It is not a substitute for monitoring real production traffic.

## Quality gate

The candidate is evaluated on the same labelled incoming batch as the champion:

- **Primary metric — macro-F1:** treats all three Iris classes equally.
- **Accuracy guardrail:** candidate accuracy must be at least **0.90**.
- **Regression guardrail:** candidate macro-F1 must be no more than **0.01** below champion macro-F1.

If both checks pass, the candidate is marked `ELIGIBLE_FOR_MANUAL_PROMOTION`. Otherwise, the decision is `RETAIN_EXISTING_CHAMPION`.

**Important:** passing the gate does not deploy the candidate. Promotion remains a deliberate manual action. The existing `models/irifulls_model.joblib` is never overwritten by this workflow.

## Generated assessment evidence

After a successful run, inspect:

| Output | Purpose |
|---|---|
| `data/incoming_iris_batch.csv` | Generated incoming labelled batch |
| `data/iris_training_data.csv` | Original training-data snapshot |
| `artifacts/monitoring/evaluation_report.json` | Data validation, class distribution, champion and candidate metrics |
| `artifacts/monitoring/candidate_model.joblib` | Separately saved candidate model |
| `artifacts/monitoring/quality_gate_decision.json` | Quality-gate thresholds, comparison, and final action |
| `mlflow_monitoring.db` | Local MLflow tracking database |

The GitHub Actions workflow uploads the monitoring outputs as an artifact named **`iris-model-monitoring-evidence`** (14-day retention).

## View MLflow experiments

After running the monitoring script locally:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow_monitoring.db --host 127.0.0.1 --port 6001
```

Open [http://127.0.0.1:6001](http://127.0.0.1:6001) in your browser.

## Run the FastAPI prediction service

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to explore the API endpoints.

## GitHub Actions

The **Iris Model Monitoring and Retraining** workflow runs on pushes affecting the monitoring code, its tests, or dependencies, and can also be started manually from the repository's **Actions** tab using **Run workflow**.

The workflow:
- installs the Python dependencies;
- generates the deterministic incoming batch;
- runs monitoring, evaluation, and candidate retraining;
- executes the tests; and
- uploads the generated assessment evidence.

See [the workflow file](.github/workflows/model-monitoring.yml) and [the detailed monitoring guide](monitoring/README.md).

## Assessment explanation

For the task-by-task mapping and rationale behind the metrics and quality gate, see [ASSESSMENT_EXPLANATION.md](ASSESSMENT_EXPLANATION.md).
