from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict():
    response = client.post("/predict/", json={"data": [5.1, 3.5, 1.4, 0.2]})
    assert response.status_code == 200

def test_predict_batch():
    response = client.post("/predict_batch/", json={"data": [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]]})
    assert response.status_code == 200
