
from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Iris Prediction API")

# Load trained model
model_path = Path(__file__).resolve().parent / "models" / "irifulls_model.joblib"
model = joblib.load(model_path)

# Iris class names
class_names = ["setosa", "versicolor", "virginica"]


# Request schema
class IrisRequest(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}


# Prediction endpoint
@app.post("/predict")
def predict_iris(request: IrisRequest):
    features = [[
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width
    ]]

    prediction = int(model.predict(features)[0])

    return {
        "prediction": prediction,
        "species": class_names[prediction]
    }

@app.get("/")
def home():
    return {"message": "Iris Prediction API is running"}