#!/usr/bin/env python3
"""
Standalone training script for 5D neural network interpolator.

Usage:
    python interpolator.py --data coursework_dataset.pkl --epochs 200 --save models/my_model.pkl
"""

import argparse
import sys
import os
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fivedreg.data import DataHandler
from fivedreg.model import ModelHandler


def train_model(data_path: str, epochs: int, save_path: str = None):
    """
    Train a 5D interpolator model.
    
    Args:
        data_path: Path to .pkl dataset
        epochs: Number of training epochs
        save_path: Path to save trained model (optional)
    """
    print(f"🚀 5D Neural Network Interpolator Training")
    print(f"=" * 60)
    print(f"Dataset: {data_path}")
    print(f"Epochs: {epochs}")
    print(f"=" * 60)
    
    # Load data
    print("\n📂 Loading dataset...")
    data_handler = DataHandler()
    X, y = data_handler.load_dataset(data_path)
    print(f"✓ Loaded {len(X)} samples with {X.shape[1]} features")
    
    # Preprocess
    print("\n🔄 Preprocessing data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_handler.preprocess_and_split()
    print(f"✓ Train: {len(X_train)} samples")
    print(f"✓ Validation: {len(X_val)} samples")
    print(f"✓ Test: {len(X_test)} samples")
    
    # Initialize model
    print(f"\n🧠 Initializing model...")
    model = ModelHandler(
        hidden_layers=[64, 32, 16],
        learning_rate=0.001,
        max_epochs=epochs
    )
    print(f"✓ Architecture: {model.hidden_layers}")
    print(f"✓ Learning rate: {model.learning_rate}")
    
    # Train with epoch callback
    print(f"\n🏋️  Training...\n")
    
    def epoch_callback(epoch, train_mse, val_mse, learning_rate):
        """Print epoch progress"""
        # Calculate validation metrics
        val_preds = model.predict(X_val)
        val_mse_actual = float(np.mean((val_preds - y_val) ** 2))
        
        print(f"Epoch {epoch:4d} | train MSE {train_mse:.5f} | val MSE {val_mse_actual:.5f} | lr {learning_rate:.2e}")
    
    import numpy as np
    start_time = time.time()
    results = model.fit(X_train, y_train, epoch_callback=epoch_callback)
    training_time = time.time() - start_time
    
    # Final evaluation
    print(f"\n" + "=" * 60)
    print(f"✓ Training complete in {training_time:.2f}s")
    
    # Test set evaluation
    test_preds = model.predict(X_test)
    test_mse = np.mean((test_preds - y_test) ** 2)
    test_rmse = np.sqrt(test_mse)
    
    # Denormalize for reporting (approximate)
    y_std = np.std(y_test)
    test_rmse_denorm = test_rmse * y_std
    
    print(f"\n📊 Test Results:")
    print(f"  RMSE (normalized target units): {test_rmse_denorm:.5f}")
    
    # Save model
    if save_path:
        print(f"\n💾 Saving model...")
        model.save_model(save_path)
        print(f"✓ Model saved to {save_path}")
    
    print(f"\n✅ Done!")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train 5D Neural Network Interpolator")
    parser.add_argument("--data", required=True, help="Path to .pkl dataset")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--save", help="Path to save trained model")
    
    args = parser.parse_args()
    
    # Generate default save path if not provided
    if not args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save = f"models/interpolator_{timestamp}.pkl"
    
    train_model(args.data, args.epochs, args.save)


if __name__ == "__main__":
    main()
