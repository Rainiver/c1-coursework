import numpy as np
import time
from typing import List, Optional, Callable
import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    """PyTorch MLP for 5D interpolation"""
    def __init__(self, in_dim=5, hidden_layers=[64, 32, 16], out_dim=1):
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
    def __init__(self, hidden_layers: List[int] = [128, 64, 32], learning_rate: float = 0.001, max_epochs: int = 500):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False, epoch_callback: Optional[Callable] = None, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        Train the model using PyTorch.
        
        Args:
            X: Training features
            y: Training targets
            verbose: Print training progress
            epoch_callback: Optional callback(epoch, train_mse, val_mse, lr) called each epoch
            X_val: Validation features for early stopping
            y_val: Validation targets for early stopping
        """
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
        
        # Training
        start_time = time.time()
        losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 50  # Early stopping patience - increased for better convergence
        
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
        """Predicts using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy().flatten()
    
    def save_model(self, filepath: str):
        """Save model to pickle file"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_data = {
            'state_dict': self.model.state_dict(),
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
        
        self.hidden_layers = model_data['hidden_layers']
        self.learning_rate = model_data['learning_rate']
        self.max_epochs = model_data['max_epochs']
        
        self.model = MLP(in_dim=5, hidden_layers=self.hidden_layers, out_dim=1).to(self.device)
        self.model.load_state_dict(model_data['state_dict'])
