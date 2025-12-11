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
    
    def preprocess_and_split(self, train_ratio=0.6, val_ratio=0.2):
        """
        Normalize and split data into train/val/test sets.
        
        Returns:
            (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        if self.X is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")
        
        # Split indices
        n_samples = len(self.X)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        # Shuffle
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        # Split
        X_train, y_train = self.X[train_idx], self.y[train_idx]
        X_val, y_val = self.X[val_idx], self.y[val_idx]
        X_test, y_test = self.X[test_idx], self.y[test_idx]
        
        # Fit scaler on training data only
        self.scaler.fit(X_train)
        
        # Transform all sets
        X_train = self.scaler.transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
