Usage Guide
===========

Detailed workflows and usage examples.

Upload Dataset
--------------

Via Web Interface
~~~~~~~~~~~~~~~~~

1. Navigate to http://localhost:3000
2. Click "Upload Dataset"
3. Select your ``.pkl`` file
4. Click "Upload"
5. Wait for confirmation

Supported format: Pickle files with ``X`` (N×5) and ``y`` (N×1) numpy arrays.

Via API
~~~~~~~

.. code-block:: bash

    curl -X POST http://localhost:8000/upload \
      -F "file=@coursework_dataset.pkl"

Training
--------

Via Web Interface
~~~~~~~~~~~~~~~~~

1. Ensure dataset is uploaded
2. Click "Train" tab
3. Configure parameters:
   * Epochs: 200 (recommended)
   * Learning rate: 0.005 (default, optimal)
   * Architecture: [256, 128, 64, 32] (default, optimal)
4. Click "Start Training"
5. Monitor progress bar
6. View results:
   * Test RMSE
   * R² Score
   * Training Time

Via API
~~~~~~~

.. code-block:: bash

    curl -X POST http://localhost:8000/train \
      -H "Content-Type: application/json" \
      -d '{
        "hidden_layers": [256, 128, 64, 32],
        "learning_rate": 0.005,
        "max_epochs": 200
      }'

Via Python
~~~~~~~~~~

.. code-block:: python

    from data import DataHandler
    from interpolator import ModelHandler
    
    # Load data
    data = DataHandler()
    data.load_dataset("coursework_dataset.pkl")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data.preprocess_and_split()
    
    # Train model
    model = ModelHandler()
    results = model.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=True)
    
    # Evaluate
    import numpy as np
    predictions = model.predict(X_test)
    rmse = np.sqrt(np.mean((predictions - y_test)**2))
    print(f"Test RMSE: {rmse:.6f}")

Making Predictions
------------------

Via Web Interface
~~~~~~~~~~~~~~~~~

1. Ensure model is trained
2. Click "Predict" tab
3. Adjust input sliders (X1-X5)
4. Click "Predict"
5. View prediction result

Via API
~~~~~~~

.. code-block:: bash

    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"inputs": [1.0, 2.0, 3.0, 4.0, 5.0]}'

Via Python
~~~~~~~~~~

.. code-block:: python

    # Assuming model is already trained
    import numpy as np
    
    X_new = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    prediction = model.predict(X_new)
    print(f"Prediction: {prediction[0][0]:.4f}")

Performance Benchmarking
------------------------

Run comprehensive performance tests:

.. code-block:: bash

    cd backend
    python scripts/benchmark_performance.py

This will test dataset sizes: 500, 1K, 2K, 3.5K, 5K, 10K samples.

Results saved to: ``performance_benchmark_results.json``

Hyperparameter Tuning
---------------------

Find optimal hyperparameters:

.. code-block:: bash

    cd backend
    python scripts/tune_hyperparameters.py

Tests 8 configurations and saves best to ``best_hyperparameters.txt``.

Docker Management
-----------------

Start services::

    ./backend/scripts/docker-start.sh

Stop services::

    ./backend/scripts/docker-stop.sh

View logs::

    ./backend/scripts/docker-logs.sh

Restart services::

    docker-compose restart

Common Workflows
----------------

Full Training Pipeline
~~~~~~~~~~~~~~~~~~~~~~

1. Upload dataset
2. Train with optimal config
3. Verify performance (RMSE < 0.03)
4. Make predictions
5. Save model (automatic)

Retraining
~~~~~~~~~~

Simply upload a new dataset and train again. Previous model is overwritten.

Batch Predictions
~~~~~~~~~~~~~~~~~

.. code-block:: python

    X_batch = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [2.0, 3.0, 4.0, 5.0, 6.0],
        [3.0, 4.0, 5.0, 6.0, 7.0],
    ])
    predictions = model.predict(X_batch)

Troubleshooting
---------------

Training fails
~~~~~~~~~~~~~~

* Ensure dataset is uploaded
* Check dataset format (X: N×5, y: N×1)
* Verify sufficient samples (>500)

Predictions are NaN
~~~~~~~~~~~~~~~~~~~

* Model may not be trained
* Input values may be out of range
* Re-train the model

Docker containers won't start
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Check ports 3000, 8000 are free
* Increase Docker memory to 4GB+
* Run ``docker-compose down`` then retry
