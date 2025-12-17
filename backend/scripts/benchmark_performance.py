#!/usr/bin/env python3
"""
Performance Benchmarking Script
Measures impact of dataset size on training performance, memory usage, and model accuracy.

System: Apple M4 (16GB RAM)
Purpose: Understand scalability and computational dynamics
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

import numpy as np
import time
import platform
import psutil
import json
from datetime import datetime
from typing import Dict, List, Tuple
import gc

from interpolator.model import ModelHandler, compute_norm_stats, apply_x_norm, apply_y_norm, generate_synthetic_pkl
from interpolator.data import DataHandler

# System Information
def get_system_info() -> Dict:
    """Get detailed system information for reproducibility"""
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "machine": platform.machine(),
    }


def measure_training_performance(n_samples: int, n_runs: int = 5) -> Dict:
    """
    Measure training performance with given dataset size
    
    Args:
        n_samples: Number of samples in dataset
        n_runs: Number of repeated runs for statistical reliability
        
    Returns:
        Dictionary with mean and std of metrics
    """
    print(f"\n{'='*70}")
    print(f"Testing with {n_samples} samples ({n_runs} runs for statistical reliability)")
    print(f"{'='*70}")
    
    metrics = {
        "training_time": [],
        "memory_peak_mb": [],
        "test_rmse": [],
        "test_r2": [],
        "epochs_trained": [],
    }
    
    for run in range(n_runs):
        print(f"  Run {run + 1}/{n_runs}...", end=" ", flush=True)
        
        # Force garbage collection
        gc.collect()
        
        # Generate synthetic data
        data_file = f"temp_data_{n_samples}.pkl"
        generate_synthetic_pkl(n=n_samples, filename=data_file)
        
        # Load and split data
        data_handler = DataHandler()
        data_handler.load_dataset(data_file)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_handler.preprocess_and_split()
        
        # Normalize
        stats = compute_norm_stats(X_train, y_train)
        X_train_norm = apply_x_norm(X_train, stats)
        y_train_norm = apply_y_norm(y_train, stats)
        X_val_norm = apply_x_norm(X_val, stats)
        y_val_norm = apply_y_norm(y_val, stats)
        X_test_norm = apply_x_norm(X_test, stats)
        
        # Initialize model
        model = ModelHandler(
            hidden_layers=[256, 128, 64, 32],
            learning_rate=5e-3,
            max_epochs=200,
            batch_size=256,
            weight_decay=1e-6
        )
        model.stats = stats
        
        # Track memory before training
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Time training
        start_time = time.perf_counter()
        results = model.fit(X_train_norm, y_train_norm, X_val=X_val_norm, y_val=y_val_norm, verbose=False)
        end_time = time.perf_counter()
        
        training_time = end_time - start_time
        
        # Track memory after training
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = mem_after - mem_before
        
        # Evaluate
        from interpolator.model import invert_y_norm
        y_pred_norm = model.predict(X_test_norm)
        y_pred = invert_y_norm(y_pred_norm, stats)
        
        test_rmse = np.sqrt(np.mean((y_pred - y_test)**2))
        ss_res = np.sum((y_test - y_pred)**2)
        ss_tot = np.sum((y_test - np.mean(y_test))**2)
        test_r2 = 1 - (ss_res / ss_tot)
        
        # Record metrics
        metrics["training_time"].append(training_time)
        metrics["memory_peak_mb"].append(memory_used)
        metrics["test_rmse"].append(test_rmse)
        metrics["test_r2"].append(test_r2)
        metrics["epochs_trained"].append(len(results["losses"]))
        
        print(f"Time: {training_time:.2f}s, RMSE: {test_rmse:.6f}")
        
        # Cleanup
        os.remove(data_file)
        gc.collect()
    
    # Calculate statistics
    summary = {}
    for key, values in metrics.items():
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values": [float(v) for v in values]
        }
    
    print(f"\n  Summary:")
    print(f"    Training Time: {summary['training_time']['mean']:.2f}s ± {summary['training_time']['std']:.2f}s")
    print(f"    Memory Used: {summary['memory_peak_mb']['mean']:.1f} ± {summary['memory_peak_mb']['std']:.1f} MB")
    print(f"    Test RMSE: {summary['test_rmse']['mean']:.6f} ± {summary['test_rmse']['std']:.6f}")
    print(f"    Test R²: {summary['test_r2']['mean']:.4f} ± {summary['test_r2']['std']:.4f}")
    
    return summary


def main():
    print("="*70)
    print("PERFORMANCE BENCHMARKING: Dataset Size Impact Analysis")
    print("="*70)
    
    # System info
    sys_info = get_system_info()
    print("\nSystem Information:")
    print(f"  Platform: {sys_info['platform']}")
    print(f"  Processor: {sys_info['processor']}")
    print(f"  CPU Cores: {sys_info['cpu_count']} physical, {sys_info['cpu_count_logical']} logical")
    print(f"  RAM: {sys_info['total_ram_gb']} GB")
    print(f"  Python: {sys_info['python_version']}")
    
    # Test different dataset sizes (including 10K as per requirement 8)
    dataset_sizes = [500, 1000, 2000, 3500, 5000, 10000]
    n_runs = 5  # Number of runs per size for statistical reliability

    
    print(f"\nTesting {len(dataset_sizes)} dataset sizes with {n_runs} runs each")
    print("This will take approximately 10-15 minutes...\n")
    
    results = {
        "system_info": sys_info,
        "timestamp": datetime.now().isoformat(),
        "n_runs_per_size": n_runs,
        "dataset_sizes": dataset_sizes,
        "measurements": {}
    }
    
    for size in dataset_sizes:
        results["measurements"][str(size)] = measure_training_performance(size, n_runs)
    
    # Save results
    output_file = "performance_benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_file}")
    
    # Generate summary report
    print("\n" + "="*70)
    print("SCALING ANALYSIS SUMMARY")
    print("="*70)
    print(f"{'Size':<10} {'Time (s)':<15} {'Memory (MB)':<15} {'RMSE':<15} {'R²':<10}")
    print("-"*70)
    
    for size in dataset_sizes:
        metrics = results["measurements"][str(size)]
        print(f"{size:<10} "
              f"{metrics['training_time']['mean']:>6.2f} ± {metrics['training_time']['std']:<5.2f} "
              f"{metrics['memory_peak_mb']['mean']:>6.1f} ± {metrics['memory_peak_mb']['std']:<5.1f} "
              f"{metrics['test_rmse']['mean']:>7.5f} ± {metrics['test_rmse']['std']:<6.5f} "
              f"{metrics['test_r2']['mean']:.4f}")
    
    # Complexity analysis
    print("\n" + "="*70)
    print("COMPUTATIONAL COMPLEXITY ANALYSIS")
    print("="*70)
    
    sizes_array = np.array(dataset_sizes)
    times_array = np.array([results["measurements"][str(s)]["training_time"]["mean"] for s in dataset_sizes])
    
    # Fit to O(n) - linear
    linear_coef = np.polyfit(sizes_array, times_array, 1)
    linear_fit = np.polyval(linear_coef, sizes_array)
    linear_r2 = 1 - np.sum((times_array - linear_fit)**2) / np.sum((times_array - np.mean(times_array))**2)
    
    print(f"Linear fit (O(n)): Time ≈ {linear_coef[0]:.4f} * n + {linear_coef[1]:.2f}")
    print(f"  R² = {linear_r2:.4f}")
    print(f"  Interpretation: Training time grows approximately linearly with dataset size")
    print(f"  Predicted time for 10,000 samples: ~{np.polyval(linear_coef, 10000):.1f}s")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
