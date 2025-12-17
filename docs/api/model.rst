Model Module
============

The ``interpolator`` module contains the neural network implementation.

.. automodule:: interpolator
   :members:
   :undoc-members:
   :show-inheritance:

ModelHandler Class
------------------

.. autoclass:: interpolator.ModelHandler
   :members:
   :special-members: __init__

Constructor
~~~~~~~~~~~

.. code-block:: python

    ModelHandler(
        hidden_layers=[256, 128, 64, 32],
        learning_rate=5e-3,
        max_epochs=200,
        batch_size=256,
        weight_decay=1e-6
    )

**Parameters:**

* ``hidden_layers`` (List[int]): Network architecture (default: [256,128,64,32])
* ``learning_rate`` (float): Adam optimizer learning rate (default: 5e-3)
* ``max_epochs`` (int): Maximum training epochs (default: 200)
* ``batch_size`` (int): Training batch size (default: 256)
* ``weight_decay`` (float): L2 regularization (default: 1e-6)

Methods
~~~~~~~

fit(X, y, X_val=None, y_val=None, verbose=False)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Train the model.

**Parameters:**

* ``X`` (np.ndarray): Training inputs (N×5)
* ``y`` (np.ndarray): Training targets (N×1)
* ``X_val`` (np.ndarray, optional): Validation inputs
* ``y_val`` (np.ndarray, optional): Validation targets
* ``verbose`` (bool): Print training progress

**Returns:**

* dict: Training results including losses and timing

predict(X)
^^^^^^^^^^

Make predictions.

**Parameters:**

* ``X`` (np.ndarray): Input data (N×5)

**Returns:**

* np.ndarray: Predictions (N×1)

MLP Class
---------

.. autoclass:: interpolator.MLP
   :members:
   :special-members: __init__

PyTorch neural network implementation.
