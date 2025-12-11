"""
Data handling utilities for 5D interpolator
"""

import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from typing import Tuple


class DataHandler:
    """Handler for loading and preprocessing 5D datasets"""
    
    def __init__(self):
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        
    def load_dataset(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load dataset from pickle file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.X = data['X']
        self.y = data['y']
        
        return self.X, self.y
    
    def preprocess_and_split(self, train_ratio=0.7, val_ratio=0.1):
        """
        Split data into train/val/test sets (70/10/20 or 3500/500/1000 for 5000 samples).
        Validation set is used for early stopping.
        
        Returns:
            (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        if self.X is None:
            raise ValueError("No data loaded")
        
        # Simple sequential split
        n = len(self.X)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        X_train = self.X[:n_train]
        y_train = self.y[:n_train]
        X_val = self.X[n_train:n_train+n_val]
        y_val = self.y[n_train:n_train+n_val]
        X_test = self.X[n_train+n_val:]
        y_test = self.y[n_train+n_val:]
        
        # Fit scaler only on training data
        self.scaler.fit(X_train)
        
        # Transform all sets
        X_train = self.scaler.transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
