import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

class DataHandler:
    def __init__(self, filepath: Optional[str] = None):
        self.scaler = StandardScaler()
        self.filepath = filepath
        self.X = None
        self.y = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

    def load_dataset(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Loads X and y from a .pkl or .npz file."""
        self.filepath = filepath
        if filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                # Handle case where keys might be different or strict structure
                if isinstance(data, dict):
                    if 'X' in data and 'y' in data:
                        self.X = data['X']
                        self.y = data['y']
                    else:
                        raise ValueError("Pickle file must contain dictionary with keys 'X' and 'y'")
                else:
                     raise ValueError("Pickle content must be a dictionary.")
        elif filepath.endswith('.npz'):
            data = np.load(filepath)
            self.X = data['X']
            self.y = data['y']
        else:
            raise ValueError("Unsupported file format. Use .pkl or .npz")
            
        return self.X, self.y

    def preprocess_and_split(self, test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42):
        """Standardizes data and splits into train/val/test."""
        if self.X is None or self.y is None:
            raise ValueError("Data not loaded. Call load_dataset first.")

        # First split into train+val and test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        # Then split train+val into train and val
        # Adjust val_size relative to the temp set
        relative_val_size = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=relative_val_size, random_state=random_state
        )

        # Standardize
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)

        return (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test)

    def get_scaler(self):
        return self.scaler
