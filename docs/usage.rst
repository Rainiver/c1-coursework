Usage Guide
===========

This guide covers detailed usage of the Fivedreg Interpolator system.

Data Handling
-------------

Supported Formats
~~~~~~~~~~~~~~~~~

The system accepts ``.pkl`` files containing:

- ``X``: NumPy array of shape ``(n_samples, 5)`` - 5 features per sample
- ``y``: NumPy array of shape ``(n_samples,)`` - target values

Example data structure::

    import pickle
    import numpy as np
    
    data = {
        'X': np.random.rand(5000, 5),
        'y': np.random.rand(5000)
    }
    
    with open('my_dataset.pkl', 'wb') as f:
        pickle.dump(data, f)

Data Preprocessing
~~~~~~~~~~~~~~~~~~

Upon upload, the system automatically:

1. **Validates** dimensions (must be 5 features)
2. **Checks** for missing values (NaN detection)
3. **Calculates** statistics (min, max, mean of target)
4. **Plans** train/val/test split (60%/20%/20%)

During training, preprocessing includes:

- **Standardization**: Features are standardized using ``StandardScaler``
- **Splitting**: Data is split into train, validation, and test sets
- **Fitting**: Scaler is fitted on training data only

Model Training
--------------

Hyperparameters
~~~~~~~~~~~~~~~

Configure the following parameters:

- **Hidden Layers**: Comma-separated layer sizes (e.g., ``64,32,16``)
  
  - Determines network architecture
  - More layers/neurons = higher capacity but slower training
  - Recommended: 2-4 layers with 16-128 neurons each

- **Learning Rate**: Float value (e.g., ``0.001``)
  
  - Controls optimization step size
  - Too high: unstable training
  - Too low: slow convergence
  - Recommended range: 0.0001 - 0.01

- **Max Epochs**: Integer (e.g., ``100``)
  
  - Maximum training iterations
  - Model may converge earlier
  - Recommended: 50-200 for typical datasets

Training Process
~~~~~~~~~~~~~~~~

1. Data is split into train (60%), validation (20%), test (20%)
2. Features are standardized using the training set statistics
3. Model trains using Adam optimizer
4. Progress is tracked via loss curve
5. Final metrics calculated on validation set

Example Training Session::

    # Via UI:
    1. Upload dataset
    2. Go to Train page
    3. Set: Hidden Layers = "128,64,32", Learning Rate = 0.0005, Max Epochs = 150
    4. Click "Start Training"
    5. Review metrics: Training Time, Final Loss, Val MSE, Val R²

Making Predictions
------------------

Input Format
~~~~~~~~~~~~

Provide exactly 5 feature values corresponding to your input dimensions.

Example via UI::

    Feature 1: 0.45
    Feature 2: 0.67
    Feature 3: 0.23
    Feature 4: 0.89
    Feature 5: 0.12

The system will:

1. Apply the same standardization used during training
2. Pass through the trained neural network
3. Return the predicted target value

Programmatic Usage (API)
--------------------------

The backend exposes REST endpoints for programmatic access.

Upload Dataset
~~~~~~~~~~~~~~

::

    POST /api/upload
    Content-Type: multipart/form-data
    
    Body: file=<dataset.pkl>
    
    Response: {
      "filename": "dataset.pkl",
      "status": "uploaded",
      "statistics": { ... }
    }

Train Model
~~~~~~~~~~~

::

    POST /api/train
    Content-Type: application/json
    
    Body: {
      "hidden_layers": [64, 32, 16],
      "learning_rate": 0.001,
      "max_epochs": 100
    }
    
    Response: {
      "status": "training_complete",
      "metrics": {
        "training_time": 0.142,
        "final_loss": 0.0012,
        "val_mse": 0.0108,
        "val_r2": 0.9745
      }
    }

Predict
~~~~~~~

::

    POST /api/predict
    Content-Type: application/json
    
    Body: {
      "features": [0.5, 0.3, 0.7, 0.2, 0.9]
    }
    
    Response: {
      "prediction": 2.456
    }

Best Practices
--------------

1. **Always upload data first** before training
2. **Start with default hyperparameters** and tune if needed
3. **Monitor validation R²**: Should be > 0.90 for good performance
4. **Use consistent feature ranges** with your training data when predicting
5. **Re-train if you upload new data** (model doesn't persist across uploads)

Limitations
-----------

- **Stateless**: Each deployment is independent (serverless on Vercel)
- **No model persistence**: Model is stored in memory only during session
- **5 features only**: Hardcoded for this specific task
- **Single user**: No multi-user support or authentication
