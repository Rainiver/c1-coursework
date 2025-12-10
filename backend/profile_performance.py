import time
import numpy as np
import torch
import psutil
import os
import sys
from fivedreg.model import ModelHandler
from fivedreg.data import DataHandler

# Mock data generation for profiling
def generate_data(n_samples):
    X = np.random.rand(n_samples, 5).astype(np.float32)
    y = np.sum(X, axis=1) + np.random.normal(0, 0.1, n_samples).astype(np.float32)
    return X, y

def profile():
    sizes = [1000, 5000, 10000]
    results = []
    
    print("Starting Performance Profiling...")
    print(f"{'Size':<10} | {'Time (s)':<10} | {'Memory (MB)':<12} | {'MSE':<10} | {'R2':<10}")
    print("-" * 65)
    
    for size in sizes:
        X, y = generate_data(size)
        
        # Memory before
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024
        
        handler = ModelHandler(max_epochs=50) # Reduced epochs for speed in demo
        
        # Train
        res = handler.fit(X, y)
        train_time = res["training_time"]
        
        # Memory after (peak approximation)
        mem_after = process.memory_info().rss / 1024 / 1024
        mem_diff = mem_after - mem_before
        
        # Eval
        preds = handler.predict(X)
        mse = np.mean((preds - y)**2)
        r2 = 1 - (np.sum((y - preds)**2) / np.sum((y - np.mean(y))**2))
        
        print(f"{size:<10} | {train_time:<10.4f} | {mem_diff:<12.2f} | {mse:<10.4f} | {r2:<10.4f}")
        
if __name__ == "__main__":
    profile()
