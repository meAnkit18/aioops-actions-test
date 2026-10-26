# Import testclient so we can send simpulated http requrests

from fastapi.testclient import TestClient
#import application object defined in main.py
from main import app

# resuable test clinet form all the test fuctions
client = TestClient(app)

def test_health_endpoint():
    """
    Verify that the application is ruuning and can respond to  a health chek request
    """
    response = client.get("/health")
    assert response.status_code ==200
    assert response.json() =={"status":"healthy"}

def test_valid_iris_prediction():
    """
    Verify that we are getting valide predictions
    """
    payload = {
    "sepal_length":3.4,
    "sepal_width":1.4,
    "petal_length":3.4,
    "petal_width":0.4
    }

    # send a simulated POST requesst with JSON input
    response = client.post("/predict", json = payload)
    #endpoint should accept the valid input
    assert response.status_code ==200
    result = response.json()
    assert "prediction" in result
    assert "species" in result

    assert result["species"] in ["setosa","vericolor", "virginica"]
    assert result["prediction"] in [0,1,2]