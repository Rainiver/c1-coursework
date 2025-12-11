import numpy as np
import time
from typing import List, Optional, Callable
from sklearn.neural_network import MLPRegressor
import pickle
import os

class ModelHandler:
    def __init__(self, hidden_layers: List[int] = [64, 32, 16], learning_rate: float = 0.001, max_epochs: int = 100):
        self.hidden_layers = tuple(hidden_layers)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False, epoch_callback: Optional[Callable] = None):
        """
        Trains the model using Scikit-Learn MLPRegressor.
        
        Args:
            X: Training features
            y: Training targets
            verbose: Print training progress
            epoch_callback: Optional callback(epoch, train_mse, val_mse, lr) called each epoch
        """
        start_time = time.time()
        
        if epoch_callback:
            # Train epoch by epoch for callbacks
            self.model = MLPRegressor(
                hidden_layer_sizes=self.hidden_layers,
                learning_rate_init=self.learning_rate,
                max_iter=1,
                random_state=42,
                solver='adam',
                activation='relu',
                verbose=verbose,
                warm_start=True
            )
            
            losses = []
            for epoch in range(self.max_epochs):
                self.model.fit(X, y)
                preds = self.model.predict(X)
                mse = float(np.mean((preds - y) ** 2))
                losses.append(mse)
                
                epoch_callback(
                    epoch=epoch + 1,
                    train_mse=mse,
                    val_mse=mse,
                    learning_rate=self.learning_rate
                )
        else:
            # Train all at once
            self.model = MLPRegressor(
                hidden_layer_sizes=self.hidden_layers,
                learning_rate_init=self.learning_rate,
                max_iter=self.max_epochs,
                random_state=42,
                solver='adam',
                activation='relu',
                verbose=verbose
            )
            
            self.model.fit(X, y)
            losses = self.model.loss_curve_
        
        training_time = time.time() - start_time
        return {"losses": losses, "training_time": training_time}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained.")
        return self.model.predict(X)
    
    def save_model(self, filepath: str):
        """Save model to pickle file"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_data = {
            'model': self.model,
            'hidden_layers': self.hidden_layers,
            'learning_rate': self.learning_rate,
            'max_epochs': self.max_epochs
        }
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load model from pickle file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.hidden_layers = model_data['hidden_layers']
        self.learning_rate = model_data['learning_rate']
        self.max_epochs = model_data['max_epochs']
