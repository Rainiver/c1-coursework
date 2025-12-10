REST API Endpoints
==================

The FastAPI backend exposes the following endpoints for web interface interaction.

Health Check
------------

**GET** ``/health`` or ``/api/health``

Check if the API server is running.

**Response**::

    {
      "status": "ok"
    }

Upload Dataset
--------------

**POST** ``/upload`` or ``/api/upload``

Upload a .pkl file containing the 5D dataset.

**Request:**
- Content-Type: ``multipart/form-data``
- Body: ``file`` - The .pkl file to upload

**Response**::

    {
      "filename": "coursework_dataset.pkl",
      "status": "uploaded",
      "statistics": {
        "total_samples": 5000,
        "n_features": 5,
        "missing_values": {
          "X": 0,
          "y": 0
        },
        "target_distribution": {
          "min": 0.1234,
          "max": 9.8765,
          "mean": 5.0123
        },
        "planned_split": {
          "train": 3000,
          "validation": 1000,
          "test": 1000
        }
      }
    }

**Error Responses:**
- 400: Invalid file format or failed to load dataset

Train Model
-----------

**POST** ``/train`` or ``/api/train``

Train the neural network interpolator on the uploaded dataset.

**Request:**
- Content-Type: ``application/json``
- Body::

    {
      "hidden_layers": [64, 32, 16],
      "learning_rate": 0.001,
      "max_epochs": 100
    }

**Response**::

    {
      "status": "training_complete",
      "metrics": {
        "training_time": 0.1420,
        "final_loss": 0.0012,
        "val_mse": 0.0108,
        "val_r2": 0.9745
      }
    }

**Error Responses:**
- 400: No dataset loaded (must upload first)
- 500: Training failed

Make Prediction
---------------

**POST** ``/predict`` or ``/api/predict``

Make a prediction using the trained model.

**Request:**
- Content-Type: ``application/json``
- Body::

    {
      "features": [0.5, 0.3, 0.7, 0.2, 0.9]
    }

**Response**::

    {
      "prediction": 2.456
    }

**Error Responses:**
- 400: Model not trained or invalid input (must have exactly 5 features)
- 500: Prediction failed

Interactive API Documentation
------------------------------

When running locally, FastAPI provides interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These interfaces allow you to:
- View all endpoints and their schemas
- Test endpoints directly in the browser
- See example requests and responses
