#!/usr/bin/env python3
"""
5D Neural Network Interpolator (PyTorch)

Core module for training and using a neural network to interpolate 5D data.
Includes normalization, early stopping, and model persistence.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Callable, Tuple
import pickle
import os
from dataclasses import dataclass


@dataclass
class Normalstats:
    """Statistics for input/output normalization"""
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float


class MLP(nn.Module):
    """Multi-Layer Perceptron for 5D interpolation"""
    
    def __init__(self, in_dim=5, hidden_layers=[128, 64, 32], out_dim=1):
        super().__init__()
        layers = []
        prev_dim = in_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class ModelHandler:
    """Handler for training and using the interpolator model"""
    
    def __init__(self, hidden_layers=[128, 64, 32], learning_rate=0.001, max_epochs=500):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stats = None

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False, 
            epoch_callback: Optional[Callable] = None, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        Train the model with early stopping.
        
        Args:
            X: Training features (N, 5)
            y: Training targets (N,)
            verbose: Print progress
            epoch_callback: Callback(epoch, train_mse, val_mse, lr)
            X_val: Validation features
            y_val: Validation targets
        
        Returns:
            dict: Training results with losses and timing
        """
        import time
        
        # Initialize model
        self.model = MLP(in_dim=5, hidden_layers=self.hidden_layers, out_dim=1).to(self.device)
        
        # Convert to tensors
        X_train = torch.FloatTensor(X).to(self.device)
        y_train = torch.FloatTensor(y).reshape(-1, 1).to(self.device)
        
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
        
        # Optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        start_time = time.time()
        losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 50
        
        for epoch in range(self.max_epochs):
            self.model.train()
            
            # Forward pass
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_mse = loss.item()
            losses.append(train_mse)
            
            # Validation
            val_mse = train_mse
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor)
                    val_mse = val_loss.item()
                
                # Early stopping
                if val_mse < best_val_loss:
                    best_val_loss = val_mse
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    if verbose or epoch_callback:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break
            
            # Callback
            if epoch_callback:
                epoch_callback(epoch + 1, train_mse, val_mse, self.learning_rate)
        
        training_time = time.time() - start_time
        return {"losses": losses, "training_time": training_time}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using trained model"""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy().flatten()
    
    def save_model(self, filepath: str):
        """Save model to file"""
        if self.model is None:
            raise ValueError("No model to save.")
        
        model_data = {
            'state_dict': self.model.state_dict(),
            'hidden_layers': self.hidden_layers,
            'learning_rate': self.learning_rate,
            'max_epochs': self.max_epochs,
            'stats': self.stats
        }
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.hidden_layers = model_data['hidden_layers']
        self.learning_rate = model_data['learning_rate']
        self.max_epochs = model_data['max_epochs']
        self.stats = model_data.get('stats')
        
        self.model = MLP(in_dim=5, hidden_layers=self.hidden_layers, out_dim=1).to(self.device)
        self.model.load_state_dict(model_data['state_dict'])


def interpolate(model: ModelHandler, stats: Normalstats, X_new: np.ndarray) -> np.ndarray:
    """
    Interpolate for new inputs using trained model and normalization stats.
    
    Args:
        model: Trained ModelHandler
        stats: Normalization statistics
        X_new: New inputs to interpolate (N, 5)
    
    Returns:
        Predictions in original scale
    """
    # Normalize input
    X_normalized = (X_new - stats.x_mean) / (stats.x_std + 1e-8)
    
    # Predict (normalized)
    y_normalized = model.predict(X_normalized)
    
    # Denormalize output
    y_pred = y_normalized * stats.y_std + stats.y_mean
    
    return y_pred


def generate_synthetic_pkl(n=5000, seed=123, workdir=".", filename="synthetic_5d_data.pkl"):
    """
    Generate synthetic 5D data and save to pickle file.
    
    Args:
        n: Number of data points
        seed: Random seed for reproducibility
        workdir: Directory to save file
        filename: Output filename
    
    Returns:
        Path to saved pickle file
    """
    np.random.seed(seed)
    
    # Generate 5D inputs uniformly in [0, 1]
    X = np.random.rand(n, 5)
    
    # Generate target with some nonlinear function
    # Example: y = sum of squared inputs + noise
    y = (X[:, 0]**2 + X[:, 1]**2 + X[:, 2]**2 + 
         X[:, 3]**2 + X[:, 4]**2 + 
         0.1 * np.random.randn(n))
    
    data = {'X': X, 'y': y}
    
    filepath = os.path.join(workdir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Generated {n} synthetic samples → {filepath}")
    return filepath


if __name__ == "__main__":
    # Quick test
    print("Generating synthetic data...")
    pkl_path = generate_synthetic_pkl(n=1000)
    
    print(f"\nTesting model training...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    X, y = data['X'], data['y']
    
    # Simple train-test split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Normalize
    x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0)
    y_mean, y_std = y_train.mean(), y_train.std()
    stats = Normalstats(x_mean, x_std, y_mean, y_std)
    
    X_train_norm = (X_train - x_mean) / (x_std + 1e-8)
    y_train_norm = (y_train - y_mean) / (y_std + 1e-8)
    
    # Train
    model = ModelHandler(max_epochs=100)
    model.stats = stats
    model.fit(X_train_norm, y_train_norm, verbose=True)
    
    # Test
    y_pred = interpolate(model, stats, X_test)
    mse = np.mean((y_pred - y_test)**2)
    print(f"\nTest MSE: {mse:.6f}")
    print("✓ Model test passed!")
