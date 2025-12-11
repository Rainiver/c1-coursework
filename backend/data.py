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
            raise ValueError("No data loaded")
        
        # Simple train-test split
        n = len(self.X)
        n_train = int(n * train_ratio)
        
        X_train = self.X[:n_train]
        y_train = self.y[:n_train]
        X_test = self.X[n_train:]
        y_test = self.y[n_train:]
        
        # Fit scaler only on training data
        self.scaler.fit(X_train)
        
        # Transform both sets
        X_train = self.scaler.transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return (X_train, y_train), (X_test, y_test)
