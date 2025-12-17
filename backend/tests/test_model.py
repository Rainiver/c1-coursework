import pytest
import numpy as np
from interpolator.model import ModelHandler


class TestModelHandler:
    """Test suite for ModelHandler class"""
    
    @pytest.fixture
    def sample_training_data(self):
        """Generate sample training data"""
        np.random.seed(42)
        X = np.random.rand(200, 5).astype(np.float32)
        y = np.sum(X, axis=1) + np.random.normal(0, 0.1, 200).astype(np.float32)
        return X, y
    
    def test_initialization_default(self):
        """Test default initialization"""
        model = ModelHandler()
        
        assert model.hidden_layers == [256, 128, 64, 32]
        assert model.learning_rate == 0.005
        assert model.max_epochs == 200
        assert model.model is None
    
    def test_initialization_custom(self):
        """Test initialization with custom parameters"""
        model = ModelHandler(
            hidden_layers=[128, 64],
            learning_rate=0.0005,
            max_epochs=50
        )
        
        assert model.hidden_layers == [128, 64]
        assert model.learning_rate == 0.0005
        assert model.max_epochs == 50
    
    def test_fit(self, sample_training_data):
        """Test model training"""
        X, y = sample_training_data
        model = ModelHandler(max_epochs=10)  # Reduced for speed
        
        results = model.fit(X, y)
        
        assert "losses" in results
        assert "training_time" in results
        assert len(results["losses"]) > 0
        assert results["training_time"] > 0
        assert model.model is not None
    
    def test_predict_after_training(self, sample_training_data):
        """Test prediction after training"""
        X, y = sample_training_data
        model = ModelHandler(max_epochs=10)
        
        model.fit(X, y)
        predictions = model.predict(X[:10])
        
        assert len(predictions) == 10
        assert predictions.shape == (10,)
        assert not np.any(np.isnan(predictions))
    
    def test_predict_without_training(self):
        """Test that prediction without training raises error"""
        model = ModelHandler()
        X = np.random.rand(10, 5)
        
        with pytest.raises(ValueError, match="Model not trained"):
            model.predict(X)
    
    def test_model_accuracy(self, sample_training_data):
        """Test that model achieves reasonable accuracy"""
        X, y = sample_training_data
        model = ModelHandler(max_epochs=50)
        
        model.fit(X, y)
        predictions = model.predict(X)
        
        mse = np.mean((predictions - y) ** 2)
        r2 = 1 - (np.sum((y - predictions)**2) / np.sum((y - np.mean(y))**2))
        
        assert mse < 1.0, "MSE should be reasonable"
        assert r2 > 0.5, "R² should be at least 0.5 on training data"
    
    def test_loss_decreases(self, sample_training_data):
        """Test that loss generally decreases during training"""
        X, y = sample_training_data
        model = ModelHandler(max_epochs=20)
        
        results = model.fit(X, y)
        losses = results["losses"]
        
        # Check that final loss is less than initial loss
        assert losses[-1] < losses[0], "Loss should decrease during training"
    
    def test_different_architectures(self, sample_training_data):
        """Test that different architectures can be trained"""
        X, y = sample_training_data
        architectures = [
            [32],
            [64, 32],
            [128, 64, 32, 16]
        ]
        
        for arch in architectures:
            model = ModelHandler(hidden_layers=arch, max_epochs=5)
            results = model.fit(X, y)
            
            assert model.model is not None
            assert len(results["losses"]) > 0
