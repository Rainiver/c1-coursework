Quick Start
===========

Get up and running in 5 minutes.

Using Docker (Fastest)
----------------------

1. **Start the application**::

    ./backend/scripts/docker-start.sh

2. **Open your browser** to http://localhost:3000

3. **Upload dataset**:

   * Click "Upload Dataset"
   * Select ``coursework_dataset.pkl``
   * Click "Upload"

4. **Train model**:

   * Go to "Train" tab
   * Set epochs: 200 (default)
   * Click "Start Training"
   * Wait ~12 seconds

5. **Make predictions**:

   * Go to "Predict" tab
   * Adjust input sliders (X1-X5)
   * Click "Predict"
   * View result

Expected Results
----------------

After training with 5000 samples:

* **Test RMSE:** ~0.025
* **R² Score:** ~0.99
* **Training Time:** 10-15 seconds

Using Python Directly
---------------------

Test the model without the web interface::

    cd backend
    python interpolator.py

This will:

* Generate 5000 synthetic samples
* Train the model
* Display metrics

Expected output::

    Test RMSE: 0.102153
    Test R² Score: 0.9761

Next Steps
----------

* Read :doc:`usage` for detailed workflows
* Check :doc:`api/index` for API reference
* See :doc:`performance` for benchmarking results
