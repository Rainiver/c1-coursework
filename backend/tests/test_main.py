import pytest
from fastapi.testclient import TestClient
import numpy as np
import os
import pickle
import sys

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from main import app, data_handler
from fivedreg.data import DataHandler
from fivedreg.model import ModelHandler

client = TestClient(app)

# Fixture for dummy data
@pytest.fixture
def dummy_pkl(tmp_path):
    d = tmp_path / "test_data.pkl"
    X = np.random.rand(100, 5).astype(np.float32)
    y = np.random.rand(100).astype(np.float32)
    with open(d, "wb") as f:
        pickle.dump({"X": X, "y": y}, f)
    return d

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_data_loading(dummy_pkl):
    dh = DataHandler()
    X, y = dh.load_dataset(str(dummy_pkl))
    assert X.shape == (100, 5)
    assert y.shape == (100,)

def test_model_training():
    X = np.random.rand(50, 5).astype(np.float32)
    y = np.random.rand(50).astype(np.float32)
    
    mh = ModelHandler(max_epochs=10)
    results = mh.fit(X, y)
    assert "losses" in results
    assert len(results["losses"]) == 10
    
    preds = mh.predict(X[:5])
    assert preds.shape == (5,)

def test_api_flow(dummy_pkl):
    # 1. Upload
    with open(dummy_pkl, "rb") as f:
        response = client.post("/upload", files={"file": ("test.pkl", f, "application/octet-stream")})
    assert response.status_code == 200
    
    # 2. Train
    response = client.post("/train", json={"hidden_layers": [10], "learning_rate": 0.01, "max_epochs": 5})
    assert response.status_code == 200
    assert response.json()["status"] == "training_complete"
    
    # 3. Predict
    response = client.post("/predict", json={"features": [0.1, 0.2, 0.3, 0.4, 0.5]})
    assert response.status_code == 200
    assert "prediction" in response.json()
