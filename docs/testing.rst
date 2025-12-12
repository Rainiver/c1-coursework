Test Suite
==========

Comprehensive testing documentation.

Overview
--------

The test suite ensures code quality and correctness through unit tests, integration tests, and end-to-end testing.

Test Structure
--------------

::

    backend/tests/
    ├── test_data.py        # DataHandler tests
    ├── test_model.py       # ModelHandler tests  
    └── test_main.py        # API endpoint tests

Running Tests
-------------

All tests
~~~~~~~~~

.. code-block:: bash

    cd backend
    pytest

With coverage
~~~~~~~~~~~~~

.. code-block:: bash

    pytest --cov=. --cov-report=html

Verbose output
~~~~~~~~~~~~~~

.. code-block:: bash

    pytest -v

Specific test file
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pytest tests/test_model.py

Test Descriptions
-----------------

test_data.py
~~~~~~~~~~~~

Tests for data handling:

* ``test_load_dataset``: Verify .pkl file loading
* ``test_preprocess_and_split``: Check train/val/test split ratios
* ``test_data_shapes``: Validate array dimensions

test_model.py
~~~~~~~~~~~~~

Tests for neural network:

* ``test_model_initialization``: Check model creation
* ``test_model_forward_pass``: Verify forward propagation
* ``test_model_training``: Test training loop
* ``test_model_prediction``: Validate predictions
* ``test_model_save_load``: Test model persistence

test_main.py
~~~~~~~~~~~~

Tests for API endpoints:

* ``test_health_endpoint``: Health check
* ``test_upload_endpoint``: File upload
* ``test_train_endpoint``: Model training
* ``test_predict_endpoint``: Predictions
* ``test_error_handling``: Error cases

Test Coverage
-------------

Current coverage: ~85%

Main areas:

* ✅ Data loading and preprocessing
* ✅ Model training and prediction
* ✅ API endpoints
* ⚠️ Error edge cases (partial)

Writing New Tests
-----------------

Example test:

.. code-block:: python

    import pytest
    import numpy as np
    from interpolator import ModelHandler
    
    def test_model_handles_small_dataset():
        model = ModelHandler()
        X = np.random.randn(100, 5)
        y = np.random.randn(100, 1)
        
        results = model.fit(X, y)
        assert 'losses' in results
        assert len(results['losses']) > 0

Continuous Integration
----------------------

Tests run automatically on:

* Git push
* Pull requests
* Pre-deployment

Expected test time: < 30 seconds

Benchmarking vs Testing
-----------------------

**Unit/Integration Tests** (pytest):
  - Fast (<1 second)
  - Test correctness
  - Run frequently

**Performance Benchmarks** (benchmark_performance.py):
  - Slow (10-15 minutes)
  - Test scalability
  - Run periodically

Debugging Failed Tests
----------------------

Run with debugging::

    pytest --pdb

Prints only failed tests::

    pytest --tb=short

Skip slow tests::

    pytest -m "not slow"
