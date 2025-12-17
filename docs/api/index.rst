API Reference
=============

Complete API documentation for all modules.

.. toctree::
   :maxdepth: 2

   data
   model
   endpoints

Core Modules
------------

:doc:`data`
    Data handling and preprocessing (DataHandler class)

:doc:`model`
    Neural network model implementation (ModelHandler, MLP)

:doc:`endpoints`
    FastAPI REST API endpoints

Quick Reference
---------------

Data Handler
~~~~~~~~~~~~

.. code-block:: python

    from data import DataHandler
    
    handler = DataHandler()
    handler.load_dataset("coursework_dataset.pkl")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = handler.preprocess_and_split()

Model Handler
~~~~~~~~~~~~~

.. code-block:: python

    from interpolator import ModelHandler
    
    model = ModelHandler(
        hidden_layers=[256, 128, 64, 32],
        learning_rate=5e-3,
        max_epochs=200
    )
    results = model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    predictions = model.predict(X_test)

API Endpoints
~~~~~~~~~~~~~

**Health Check**::

    GET /health

**Upload Dataset**::

    POST /upload
    Content-Type: multipart/form-data

**Train Model**::

    POST /train
    Content-Type: application/json
    Body: {"hidden_layers": [256,128,64,32], "learning_rate": 0.005, "max_epochs": 200}

**Make Prediction**::

    POST /predict
    Content-Type: application/json
    Body: {"inputs": [1.0, 2.0, 3.0, 4.0, 5.0]}
