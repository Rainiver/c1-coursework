Data Module
===========

The ``data`` module handles dataset loading, preprocessing, and splitting.

.. automodule:: data
   :members:
   :undoc-members:
   :show-inheritance:

DataHandler Class
-----------------

.. autoclass:: data.DataHandler
   :members:
   :special-members: __init__

Methods
~~~~~~~

load_dataset(filepath)
^^^^^^^^^^^^^^^^^^^^^^

Load a .pkl dataset file.

**Parameters:**

* ``filepath`` (str): Path to .pkl file

**Example:**

.. code-block:: python

    handler = DataHandler()
    handler.load_dataset("coursework_dataset.pkl")

preprocess_and_split(train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Split data into train/validation/test sets.

**Parameters:**

* ``train_ratio`` (float): Training set proportion (default: 0.7)
* ``val_ratio`` (float): Validation set proportion (default: 0.1)
* ``test_ratio`` (float): Test set proportion (default: 0.2)

**Returns:**

* Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test))

**Example:**

.. code-block:: python

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = handler.preprocess_and_split()
    print(f"Training samples: {len(X_train)}")  # 3500 for 5000 total
