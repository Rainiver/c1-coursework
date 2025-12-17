Performance & Profiling
=======================

This section presents a quantitative and qualitative analysis of the 5D Interpolator's performance. The goal is to characterize how the system scales with increasing data volume and to identify the computational cost of training and inference.

Experimental Setup
----------------

All benchmarks were conducted on a **MacBook Air (Apple Silicon)** running macOS. The testing environment utilized Python 3.9 and PyTorch, restricted to CPU execution to simulate a standard deployment environment.

Time measurement was performed using Python's high-precision ``time.perf_counter()`` to capture exclusive execution time, while memory usage was profiled using the ``tracemalloc`` standard library to track peak memory allocation during the training loop.

Computational Scaling Analysis
---------------------------

We investigated the relationship between dataset size and training duration by varying the number of samples ($N$) from 500 to 10,000. 

**Observation:**
The training time exhibits a distinct **linear scaling relationship** ($O(N)$). 
For a dataset of 1,000 samples, training completes in approximately **0.13 seconds**. As we scale to 10,000 samples, the time rises proportionally to **~1.14 seconds**.

**Interpretation:**
This linear behavior is expected for this Neural Network architecture. The dominant computational cost in each epoch is the forward and backward pass (backpropagation). Since the batch size processes the entire dataset (or large mini-batches), the number of floating-point operations (FLOPs) increases linearly with the number of input samples. The system demonstrates excellent efficiency, processing **~8,700 samples per second**, indicating that it can comfortably handle significantly larger datasets without requiring GPU acceleration.

Memory Dynamics
-------------

Memory profiling revealed a distinct efficient characteristic of the system.

- **Training Phase:** Peak memory usage showed negligible growth relative to dataset size for the tested range (peaking under 1MB relative to baseline for 10k samples). This suggests PyTorch's dynamic computational graph is efficiently managing tensor allocation, and the dataset sizes are well within the L1/L2 cache or fast RAM, preventing expensive swap operations.
  
- **Prediction Phase:** Inference is computationally inexpensive. The memory footprint for prediction is constant ($O(1)$) regardless of training set size, as it only depends on the fixed model architecture (weights & biases). This makes the deployed model highly scalable for high-throughput query environments.

Accuracy & Trade-offs
-------------------

We analyzed the Root Mean Square Error (RMSE) across different dataset sizes.

Interestingly, increasing data from 1,000 to 10,000 samples yielded **diminishing returns** in accuracy (dominating around RMSE ~0.10). This plateau suggests that the model capacity (4 hidden layers) has saturated the learnable patterns of the synthetic function, or that the remaining error represents the irreducible noise floor of the data generation process.

**Conclusion:**
For this specific 5D interpolation task, a dataset of **1,000 - 2,000 samples** represents the "sweet spot"—offering the optimal balance between rapid training (<0.2s) and high accuracy ($R^2 > 0.97$).
