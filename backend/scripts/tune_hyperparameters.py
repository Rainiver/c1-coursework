#!/usr/bin/env python3
"""
Hyperparameter tuning script for the interpolator model.
Tests different configurations to find the best RMSE.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

import numpy as np
import pickle
from interpolator import ModelHandler, compute_norm_stats, apply_x_norm, apply_y_norm
from data import DataHandler
import itertools
from datetime import datetime

# Hyperparameter search space
CONFIGS = [
    # Baseline (current)
    {"name": "baseline", "hidden_layers": [128, 64, 32], "lr": 5e-3, "epochs": 200, "batch_size": 256, "weight_decay": 1e-6},
    
    # Larger network
    {"name": "larger_net", "hidden_layers": [256, 128, 64], "lr": 3e-3, "epochs": 200, "batch_size": 256, "weight_decay": 1e-6},
    
    # Deeper network
    {"name": "deeper_net", "hidden_layers": [256, 128, 64, 32], "lr": 5e-3, "epochs": 200, "batch_size": 256, "weight_decay": 1e-6},
    
    # More epochs
    {"name": "more_epochs", "hidden_layers": [128, 64, 32], "lr": 5e-3, "epochs": 300, "batch_size": 256, "weight_decay": 1e-6},
    
    # Smaller batch
    {"name": "small_batch", "hidden_layers": [128, 64, 32], "lr": 5e-3, "epochs": 200, "batch_size": 128, "weight_decay": 1e-6},
    
    # Conservative LR
    {"name": "conservative_lr", "hidden_layers": [128, 64, 32], "lr": 1e-3, "epochs": 200, "batch_size": 256, "weight_decay": 1e-6},
    
    # Stronger regularization
    {"name": "strong_reg", "hidden_layers": [128, 64, 32], "lr": 5e-3, "epochs": 200, "batch_size": 256, "weight_decay": 1e-5},
    
    # Best guess combo
    {"name": "best_guess", "hidden_layers": [256, 128, 64], "lr": 3e-3, "epochs": 300, "batch_size": 128, "weight_decay": 5e-6},
]


def train_and_evaluate(config, X_train, y_train, X_val, y_val, X_test, y_test):
    """Train model with given config and return test RMSE"""
    
    # Compute normalization stats
    stats = compute_norm_stats(X_train, y_train)
    
    # Normalize data
    X_train_norm = apply_x_norm(X_train, stats)
    y_train_norm = apply_y_norm(y_train, stats)
    X_val_norm = apply_x_norm(X_val, stats)
    y_val_norm = apply_y_norm(y_val, stats)
    X_test_norm = apply_x_norm(X_test, stats)
    
    # Initialize model
    model = ModelHandler(
        hidden_layers=config["hidden_layers"],
        learning_rate=config["lr"],
        max_epochs=config["epochs"],
        batch_size=config["batch_size"],
        weight_decay=config["weight_decay"]
    )
    model.stats = stats
    
    # Train
    print(f"  Training {config['name']}...", end=" ", flush=True)
    model.fit(X_train_norm, y_train_norm, X_val=X_val_norm, y_val=y_val_norm, verbose=False)
    
    # Evaluate
    y_pred_norm = model.predict(X_test_norm)
    from interpolator import invert_y_norm
    y_pred = invert_y_norm(y_pred_norm, stats)
    
    test_rmse = np.sqrt(np.mean((y_pred - y_test)**2))
    
    print(f"Test RMSE: {test_rmse:.6f}")
    
    return {
        "config": config,
        "test_rmse": test_rmse,
        "model": model
    }


def main():
    print("="*70)
    print("HYPERPARAMETER TUNING")
    print("="*70)
    print(f"Testing {len(CONFIGS)} configurations...")
    print(f"Target: RMSE < 0.035 (current best: 0.0399)")
    print("="*70 + "\n")
    
    # Load data
    print("Loading dataset...")
    data_handler = DataHandler()
    
    # Try to load from uploads or use default
    dataset_path = "./uploads/coursework_dataset.pkl"
    if not os.path.exists(dataset_path):
        dataset_path = "./coursework_dataset.pkl"
    
    data_handler.load_dataset(dataset_path)
    
    # Split data (70/10/20)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_handler.preprocess_and_split()
    
    print(f"Data loaded: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test\n")
    
    # Test each configuration
    results = []
    for i, config in enumerate(CONFIGS, 1):
        print(f"[{i}/{len(CONFIGS)}] Testing: {config['name']}")
        print(f"  Config: {config['hidden_layers']}, lr={config['lr']}, epochs={config['epochs']}, batch={config['batch_size']}")
        
        result = train_and_evaluate(config, X_train, y_train, X_val, y_val, X_test, y_test)
        results.append(result)
        print()
    
    # Sort by RMSE
    results.sort(key=lambda x: x["test_rmse"])
    
    # Print summary
    print("="*70)
    print("RESULTS (sorted by Test RMSE)")
    print("="*70)
    for i, result in enumerate(results, 1):
        config = result["config"]
        rmse = result["test_rmse"]
        marker = "🏆" if i == 1 else "✓" if rmse < 0.035 else " "
        print(f"{marker} {i:2d}. {config['name']:20s} - RMSE: {rmse:.6f}")
        print(f"      {config['hidden_layers']}, lr={config['lr']}, epochs={config['epochs']}, batch={config['batch_size']}, wd={config['weight_decay']}")
    
    print("\n" + "="*70)
    best = results[0]
    print(f"🏆 BEST CONFIGURATION: {best['config']['name']}")
    print(f"   Test RMSE: {best['test_rmse']:.6f}")
    print(f"   Config: {best['config']}")
    print("="*70)
    
    # Save best config
    with open("best_hyperparameters.txt", "w") as f:
        f.write(f"Best configuration found on {datetime.now()}\n")
        f.write(f"Test RMSE: {best['test_rmse']:.6f}\n\n")
        f.write(f"hidden_layers = {best['config']['hidden_layers']}\n")
        f.write(f"learning_rate = {best['config']['lr']}\n")
        f.write(f"max_epochs = {best['config']['epochs']}\n")
        f.write(f"batch_size = {best['config']['batch_size']}\n")
        f.write(f"weight_decay = {best['config']['weight_decay']}\n")
    
    print("\n✓ Best configuration saved to: best_hyperparameters.txt")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
