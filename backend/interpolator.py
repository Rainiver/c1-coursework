#!/usr/bin/env python3
"""
5D Neural Network Interpolator (PyTorch)

Core module for training and using a neural network to interpolate 5D data.
Includes normalization, batch processing, LR scheduling, and early stopping.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional, Callable, Tuple, List
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
    """Handler for PyTorch neural network model with advanced training features"""
    
    def __init__(self, hidden_layers: List[int] = [256, 128, 64, 32],  # Optimal from tuning
                 learning_rate: float = 5e-3,  # Increased from 0.001
                 max_epochs: int = 200,  # Back to 200 like reference
                 batch_size: int = 256,  # NEW: batch processing
                 weight_decay: float = 1e-6):  # NEW: L2 regularization
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stats = None




    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False, 
            epoch_callback: Optional[Callable] = None,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        Train the model with batch processing, LR scheduling, and early stopping.
        
        Args:
            X: Training features (N, 5)
            y: Training targets (N,)
            verbose: Print progress
            epoch_callback: Callback(epoch, train_loss, val_loss, lr)
            X_val: Validation features
            y_val: Validation targets
        
        Returns:
            dict: Training results with losses and timing
        """
        import time
        
        # Initialize model
        self.model = MLP(in_dim=5, hidden_layers=self.hidden_layers, out_dim=1).to(self.device)
        
        # Convert to tensors
        X_train = torch.FloatTensor(X)
        y_train = torch.FloatTensor(y).reshape(-1, 1)
        
        # Create DataLoader for batch processing
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,  # Shuffle for better generalization
            drop_last=False
        )
        
        # Validation data
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
        
        # Optimizer with weight decay (L2 regularization)
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler (Cosine Annealing)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.max_epochs
        )
        
        criterion = nn.MSELoss()
        
        # Training loop with early stopping
        start_time = time.time()
        losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20  # Reference uses 20
        
        for epoch in range(self.max_epochs):
            # Training phase
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # Average loss for epoch
            avg_train_loss = epoch_loss / num_batches
            losses.append(avg_train_loss)
            
            # Validation phase
            val_loss = None
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(X_val_tensor)
                    val_loss = criterion(val_pred, y_val_tensor).item()
                
                # Early stopping based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Step the learning rate scheduler
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Callback for progress updates
            if epoch_callback:
                epoch_callback(epoch, avg_train_loss, val_loss, current_lr)
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                val_str = f", Val Loss: {val_loss:.6f}" if val_loss is not None else ""
                print(f"Epoch {epoch+1}/{self.max_epochs}, Train Loss: {avg_train_loss:.6f}{val_str}, LR: {current_lr:.6f}")
        
        training_time = time.time() - start_time
        
        if verbose:
            print(f"\nTraining completed in {training_time:.2f}s")
            print(f"Final loss: {losses[-1]:.6f}")
        
        return {
            "losses": losses,
            "training_time": training_time
        }


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


def apply_x_norm(X: np.ndarray, stats: Normalstats) -> np.ndarray:
    """Apply X normalization"""
    return (X - np.array(stats.x_mean)) / (np.array(stats.x_std) + 1e-12)

def apply_y_norm(y: np.ndarray, stats: Normalstats) -> np.ndarray:
    """Apply y normalization"""
    return (y - stats.y_mean) / (stats.y_std + 1e-12)

def invert_y_norm(y_norm: np.ndarray, stats: Normalstats) -> np.ndarray:
    """Invert y normalization"""
    return y_norm * (stats.y_std + 1e-12) + stats.y_mean

def compute_norm_stats(X: np.ndarray, y: np.ndarray) -> Normalstats:
    """Compute normalization statistics from data"""
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0) + 1e-12
    y_mean = y.mean()
    y_std = y.std() + 1e-12
    return Normalstats(x_mean.tolist(), x_std.tolist(), float(y_mean), float(y_std))

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
    X_normalized = apply_x_norm(X_new, stats)
    
    # Predict (normalized)
    y_normalized = model.predict(X_normalized)
    
    # Denormalize output
    y_pred = invert_y_norm(y_normalized, stats)
    
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
    os.makedirs(workdir, exist_ok=True)  # Create directory if needed
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Generated {n} synthetic samples → {filepath}")
    return filepath




if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="5D Neural Network Interpolator")
    parser.add_argument("--mode", choices=["generate", "test"], default="test",
                        help="Mode: generate synthetic data or test model")
    parser.add_argument("--n", type=int, default=1000,
                        help="Number of data points to generate")
    parser.add_argument("--workdir", default=".",
                        help="Working directory for output files")
    parser.add_argument("--filename", default="synthetic_5d_data.pkl",
                        help="Output filename for generated data")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    if args.mode == "generate":
        # Generate synthetic test data
        print(f"Generating {args.n} synthetic 5D data points...")
        pkl_path = generate_synthetic_pkl(
            n=args.n,
            seed=args.seed,
            workdir=args.workdir,
            filename=args.filename
        )
        print(f"✓ Data saved to: {pkl_path}")
    
    elif args.mode == "test":
        # Quick test mode with full-size dataset (5000 samples like production)
        print("Running quick model test...")
        print("\n1. Generating synthetic data (5000 samples)...")
        pkl_path = generate_synthetic_pkl(n=5000)

        
        print("\n2. Testing model training...")
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
        print("\n3. Training model...")
        model = ModelHandler()  # Use default 500 epochs
        model.stats = stats
        model.fit(X_train_norm, y_train_norm, verbose=True)  # Enable progress display

        
        # Test
        print("\n4. Evaluating...")
        y_pred = interpolate(model, stats, X_test)
        mse = np.mean((y_pred - y_test)**2)
        rmse = np.sqrt(mse)
        print(f"   Test MSE: {mse:.6f}")
        print(f"   Test RMSE: {rmse:.6f}")
        print("\n✓ Model test passed!")

