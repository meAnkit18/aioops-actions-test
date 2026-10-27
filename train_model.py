from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
import mlflow.sklearn
#client API lets us manage registerd model and theier aliases
from mlflow import MlflowClient

# configure mlflow tracking
# use localhost instead of 0.0.0.0 because browsers block 0.0.0.0 as a cross-origin address
mlflow.set_tracking_uri("http://127.0.0.1:6001")
# group related training runs under one experiment
mlflow.set_experiment("iris-classification")

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
n_estimators = 100
random_state = 42
model_parameters = {
    "model_type": "RandomForestClassifier",
    "n_estimators": n_estimators,
    "random_state": random_state,
    "test_size": 0.2,
}

# start a tracking training run
with mlflow.start_run() as run:
    # record the training configuration before model fitting
    mlflow.log_params(model_parameters)

    # Build classifier using the selected parameters
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
    )
    # fit the model on training data
    model.fit(X_train, y_train)

    # generate prediction using the test data
    y_pred = model.predict(X_test)

    # calculate metrics that help us evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)

    # log the evaluated results so that different runs can be compared
    mlflow.log_metrics({
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
    })

    print("MLFlow run ID: ", run.info.run_id)
    print("Test_accuracy: ", accuracy)
    print("test_precision: ", precision)
    print("test_recall: ", recall)

    model_dir = Path(__file__).resolve().parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "irifulls_model.joblib"
    joblib.dump(model, model_path)

    print(f"Model saved to: {model_path}")

    mlflow.sklearn.log_model(
        sk_model=model,
        name="iris-model",
        registered_model_name="iris-classifier",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )
    # assiggn the "champion" alias to the newly registered model
    client - MlflowClient()
    current_run_id = mlflow.active_run().info.run_id

    #Redtrieve all versions registered under this model name
    registered_versions= client.search_model_version(
        "name='iris-classifier'"
    )
    #find the model version created by the traning run 
    current_model_version= next(
        (
            version
            for version in registered_versions
            if version.run_id == current_run_id
        ),
        None
    )

    if current_model_version is None:
        raise RuntimeError(
            "Model was loogedm but its registered veriosn could not be found"
        )

    client.set_registered_alisas(
        name="iris-classifier",
        alisas="champion",
        version=current_model_version
    )
    print(f"Chamion alias assigned to model version")
    print("registered model: iris-classifier")

# Train model
# model = RandomForestClassifier(random_state=42)
# model.fit(X_train, y_train)

# # Evaluate model
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.4f}")

# # Save model
# model_dir = Path(__file__).resolve().parent / "models"
# model_dir.mkdir(parents=True, exist_ok=True)

# model_path = model_dir / "irifulls_model.joblib"
# joblib.dump(model, model_path)

# print(f"Model saved to: {model_path}")
