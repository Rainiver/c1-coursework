REST API Endpoints
==================

FastAPI backend endpoints.

Base URL
--------

* Local: ``http://localhost:8000``
* Docker: ``http://backend:8000``

Endpoints
---------

GET /health
~~~~~~~~~~~

Health check endpoint.

**Response:**

.. code-block:: json

    {
      "status": "ok"
    }

POST /upload
~~~~~~~~~~~~

Upload a dataset file (.pkl format).

**Request:**

* Content-Type: ``multipart/form-data``
* Field: ``file`` (File)

**Response:**

.. code-block:: json

    {
      "filename": "coursework_dataset.pkl",
      "samples": 5000,
      "features": 5
    }

POST /train
~~~~~~~~~~~

Train the model.

**Request Body:**

.. code-block:: json

    {
      "hidden_layers": [256, 128, 64, 32],
      "learning_rate": 0.005,
      "max_epochs": 200
    }

**Response:**

.. code-block:: json

    {
      "status": "training_complete",
      "metrics": {
        "training_time": 12.58,
        "test_rmse": 0.024635,
        "test_r2": 0.9946,
        "training_samples": 3500,
        "test_samples": 1000
      }
    }

POST /predict
~~~~~~~~~~~~~

Make a prediction.

**Request Body:**

.. code-block:: json

    {
      "inputs": [1.0, 2.0, 3.0, 4.0, 5.0]
    }

**Response:**

.. code-block:: json

    {
      "prediction": 42.15,
      "inputs": [1.0, 2.0, 3.0, 4.0, 5.0]
    }

Error Responses
---------------

All endpoints return standard HTTP status codes:

* ``200 OK``: Success
* ``400 Bad Request``: Invalid input
* ``500 Internal Server Error``: Server error

Example error response:

.. code-block:: json

    {
      "detail": "No dataset loaded. Upload first."
    }
