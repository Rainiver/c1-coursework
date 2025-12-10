Testing
=======

The Fivedreg Interpolator includes a comprehensive testing suite to ensure reliability and correctness.

Test Organization
-----------------

Tests are located in ``backend/tests/`` and organized into three files:

- ``test_data.py``: Tests for data loading, preprocessing, and validation
- ``test_model.py``: Tests for model training and prediction
- ``test_main.py``: Tests for FastAPI endpoints

Running Tests
-------------

All tests::

    cd backend
    pytest tests/ -v

Specific test file::

    pytest tests/test_data.py -v

With coverage report::

    pytest tests/ --cov=fivedreg --cov-report=html

Test Coverage
-------------

Current test coverage includes:

Data Module (``test_data.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ✓ Loading .pkl files
- ✓ Validating data dimensions
- ✓ Handling missing values
- ✓ Train/val/test splitting
- ✓ Feature standardization
- ✓ Error handling for invalid files

Model Module (``test_model.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ✓ Model initialization with custom configuration
- ✓ Training on synthetic data
- ✓ Prediction accuracy
- ✓ Loss tracking
- ✓ Error handling for untrained model

API Endpoints (``test_main.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ✓ Health check endpoint
- ✓ Upload endpoint (valid and invalid files)
- ✓ Train endpoint (with and without data)
- ✓ Predict endpoint (with and without trained model)
- ✓ Error responses and status codes

Writing New Tests
-----------------

Follow the existing test structure. Example::

    import pytest
    from fivedreg.data import DataHandler
    import numpy as np
    
    def test_load_valid_dataset():
        \"\"\"Test loading a valid dataset\"\"\"
        handler = DataHandler()
        X, y = handler.load_dataset('path/to/dataset.pkl')
        
        assert X.shape[1] == 5  # 5 features
        assert len(X) == len(y)  # Same number of samples
        assert not np.any(np.isnan(X))  # No missing values

Continuous Integration
----------------------

Tests should pass before committing code. Run tests locally first::

    pytest tests/ -v

Known Limitations
-----------------

- Tests use synthetic data for model training (not the actual coursework dataset)
- Performance tests are not included in the main test suite (use ``profile_performance.py`` separately)
- Frontend tests are not included (focus is on backend Python code)

Troubleshooting
---------------

**Import errors:**
- Ensure backend is installed: ``pip install -e ./backend``
- Check that you're in the correct directory

**Test failures:**
- Check that all dependencies are installed
- Verify Python version is 3.9+
- Look for specific error messages in test output
