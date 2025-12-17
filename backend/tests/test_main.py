import pytest
from fastapi.testclient import TestClient
import numpy as np
import os
import pickle
import sys

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from main import app, data_handler
from interpolator.data import DataHandler
from interpolator.model import ModelHandler

client = TestClient(app)

import pytest
from fastapi.testclient import TestClient
import sys
import os
import tempfile
import pickle
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import main as api_main
from main import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_globals():
    """Reset global state before each test"""
    api_main.data_handler.X = None
    api_main.data_handler.y = None
    # Access recursively in case of rebinding
    if hasattr(api_main.model_handler, 'model'):
        api_main.model_handler.model = None
    yield


class TestHealthEndpoint:
    """Tests for health check endpoint"""
    
    def test_health_check(self):
        """Test health check returns ok"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    



class TestUploadEndpoint:
    """Tests for upload endpoint"""
    
    @pytest.fixture
    def valid_pkl_file(self):
        """Create a valid .pkl file for testing"""
        X = np.random.rand(100, 5).astype(np.float32)
        y = np.random.rand(100).astype(np.float32)
        data = {'X': X, 'y': y}
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            pickle.dump(data, f)
            filepath = f.name
        
        yield filepath
        
        if os.path.exists(filepath):
            os.remove(filepath)
    
    def test_upload_valid_pkl(self, valid_pkl_file):
        """Test uploading a valid .pkl file"""
        with open(valid_pkl_file, 'rb') as f:
            response = client.post(
                "/upload",
                files={"file": ("test.pkl", f, "application/octet-stream")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "uploaded"
        assert "statistics" in data
        assert data["statistics"]["total_samples"] == 100
        assert data["statistics"]["n_features"] == 5
    
    def test_upload_invalid_extension(self):
        """Test uploading file with invalid extension"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("invalid data")
            filepath = f.name
        
        try:
            with open(filepath, 'rb') as f:
                response = client.post(
                    "/upload",
                    files={"file": ("test.txt", f, "text/plain")}
                )
            
            assert response.status_code == 400
            assert "Invalid file format" in response.json()["detail"]
        finally:
            os.remove(filepath)
    



class TestTrainEndpoint:
    """Tests for train endpoint"""
    
    @pytest.fixture
    def uploaded_dataset(self):
        """Upload a dataset before training"""
        X = np.random.rand(100, 5).astype(np.float32)
        y = np.sum(X, axis=1).astype(np.float32)
        data = {'X': X, 'y': y}
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            pickle.dump(data, f)
            filepath = f.name
        
        with open(filepath, 'rb') as f:
            client.post(
                "/upload",
                files={"file": ("test.pkl", f, "application/octet-stream")}
            )
        
        yield
        
        if os.path.exists(filepath):
            os.remove(filepath)
    
    def test_train_without_data(self):
        """Test training without uploading data first"""
        response = client.post(
            "/train",
            json={
                "hidden_layers": [64, 32, 16],
                "learning_rate": 0.001,
                "max_epochs": 10
            }
        )
        
        assert response.status_code == 400
        assert "No dataset loaded" in response.json()["detail"]
    
    def test_train_with_data(self, uploaded_dataset):
        """Test training with uploaded data"""
        response = client.post(
            "/train",
            json={
                "hidden_layers": [64, 32, 16],
                "learning_rate": 0.001,
                "max_epochs": 10
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "training_complete"
        assert "metrics" in data
        assert "training_time" in data["metrics"]
        assert "test_r2" in data["metrics"]
        assert data["metrics"]["test_r2"] is not None


class TestPredictEndpoint:
    """Tests for predict endpoint"""
    
    @pytest.fixture
    def trained_model(self):
        """Upload dataset and train model"""
        X = np.random.rand(100, 5).astype(np.float32)
        y = np.sum(X, axis=1).astype(np.float32)
        data = {'X': X, 'y': y}
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            pickle.dump(data, f)
            filepath = f.name
        
        with open(filepath, 'rb') as f:
            client.post(
                "/upload",
                files={"file": ("test.pkl", f, "application/octet-stream")}
            )
        
        client.post(
            "/train",
            json={
                "hidden_layers": [32],
                "learning_rate": 0.001,
                "max_epochs": 5
            }
        )
        
        yield
        
        if os.path.exists(filepath):
            os.remove(filepath)
    
    def test_predict_without_model(self):
        """Test prediction without training model first"""
        api_main.model_handler.model = None  # Force reset dynamically
        response = client.post(
            "/predict",
            json={"features": [0.5, 0.5, 0.5, 0.5, 0.5]}
        )
        
        assert response.status_code == 400
        assert "Model not trained" in response.json()["detail"]
    
    def test_predict_with_model(self, trained_model):
        """Test prediction with trained model"""
        response = client.post(
            "/predict",
            json={"features": [0.5, 0.5, 0.5, 0.5, 0.5]}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert isinstance(data["prediction"], (int, float))
    
    def test_predict_wrong_feature_count(self, trained_model):
        """Test prediction with wrong number of features"""
        response = client.post(
            "/predict",
            json={"features": [0.5, 0.5, 0.5]}  # Only 3 features instead of 5
        )
        
        assert response.status_code == 400
        assert "Input must have 5 features" in response.json()["detail"]

# Fixture for dummy data
@pytest.fixture
def dummy_pkl(tmp_path):
    d = tmp_path / "test_data.pkl"
    X = np.random.rand(100, 5).astype(np.float32)
    y = np.random.rand(100).astype(np.float32)
    with open(d, "wb") as f:
        pickle.dump({"X": X, "y": y}, f)
    return d

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
