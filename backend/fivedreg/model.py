import numpy as np
import time
from typing import List, Optional
from sklearn.neural_network import MLPRegressor
import pickle

class ModelHandler:
    def __init__(self, hidden_layers: List[int] = [64, 32, 16], learning_rate: float = 0.001, max_epochs: int = 100):
        self.hidden_layers = tuple(hidden_layers)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False):
        """Trains the model using Scikit-Learn MLPRegressor."""
        start_time = time.time()
        
        # MLPRegressor uses 'adam' by default. 'learning_rate_init' corresponds to learning_rate.
        # 'max_iter' corresponds to max_epochs.
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
        
        training_time = time.time() - start_time
        
        # Scikit-learn doesn't return per-epoch loss history easily in exact same format as manual loop,
        # but provides loss_curve_
        losses = self.model.loss_curve_
        
        return {"losses": losses, "training_time": training_time}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        return self.model.predict(X)
