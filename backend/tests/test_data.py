import pytest
import numpy as np
import os
import tempfile
import pickle
from interpolator.data import DataHandler


class TestDataHandler:
    """Test suite for DataHandler class"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample 5D dataset"""
        np.random.seed(42)
        X = np.random.rand(100, 5).astype(np.float32)
        y = np.sum(X, axis=1) + np.random.normal(0, 0.1, 100).astype(np.float32)
        return X, y
    
    @pytest.fixture
    def temp_pkl_file(self, sample_data):
        """Create temporary .pkl file with sample data"""
        X, y = sample_data
        data = {'X': X, 'y': y}
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            pickle.dump(data, f)
            filepath = f.name
        
        yield filepath
        
        # Cleanup
        if os.path.exists(filepath):
            os.remove(filepath)
    
    def test_initialization(self):
        """Test DataHandler can be initialized"""
        handler = DataHandler()
        assert handler.X is None
        assert handler.y is None
        assert handler.scaler is not None
    
    def test_load_pkl_dataset(self, temp_pkl_file):
        """Test loading a valid .pkl dataset"""
        handler = DataHandler()
        X, y = handler.load_dataset(temp_pkl_file)
        
        assert X.shape[1] == 5, "Should have 5 features"
        assert len(X) == len(y), "X and y should have same number of samples"
        assert X.shape[0] == 100, "Should have 100 samples"
    
    def test_load_invalid_format(self):
        """Test that loading invalid file format raises error"""
        handler = DataHandler()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("dummy data")
            filepath = f.name
            
        try:
            with pytest.raises(Exception):
                handler.load_dataset(filepath)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    def test_load_missing_file(self):
        """Test that loading non-existent file raises error"""
        handler = DataHandler()
        
        with pytest.raises(FileNotFoundError):
            handler.load_dataset("nonexistent.pkl")
    
    def test_preprocess_and_split(self, temp_pkl_file):
        """Test data preprocessing and splitting"""
        handler = DataHandler()
        handler.load_dataset(temp_pkl_file)
        
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = handler.preprocess_and_split()
        
        # Check split sizes (default 70/10/20)
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == 100, "Total samples should match original"
        assert len(X_train) == 70, "Train should be 70%"
        assert len(X_val) == 10, "Val should be 10%"
        assert len(X_test) == 20, "Test should be 20%"
        
        # Check standardization
        assert np.abs(np.mean(X_train)) < 0.3, "Training data should be standardized (mean ~0)"
        assert np.abs(np.std(X_train) - 1.0) < 0.3, "Training data should be standardized (std ~1)"
    
    def test_missing_values_detection(self):
        """Test detection of missing values"""
        X_with_nan = np.random.rand(50, 5)
        X_with_nan[0, 0] = np.nan
        y = np.random.rand(50)
        
        data = {'X': X_with_nan, 'y': y}
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            pickle.dump(data, f)
            filepath = f.name
        
        try:
            handler = DataHandler()
            X, y = handler.load_dataset(filepath)
            
            # Check that NaN is present
            assert np.any(np.isnan(X)), "Should detect NaN values"
        finally:
            os.remove(filepath)
    
    def test_get_scaler(self, temp_pkl_file):
        """Test getting the fitted scaler"""
        handler = DataHandler()
        handler.load_dataset(temp_pkl_file)
        handler.preprocess_and_split()
        
        scaler = handler.scaler
        assert scaler is not None
        assert hasattr(scaler, 'mean_')
        assert hasattr(scaler, 'scale_')
