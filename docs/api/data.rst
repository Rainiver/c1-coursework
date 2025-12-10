Data Module
===========

.. automodule:: fivedreg.data
   :members:
   :undoc-members:
   :show-inheritance:

DataHandler Class
-----------------

.. autoclass:: fivedreg.data.DataHandler
   :members:
   :undoc-members:
   :show-inheritance:
   
   .. automethod:: __init__
   .. automethod:: load_dataset
   .. automethod:: preprocess_and_split
   .. automethod:: get_scaler

Usage Example
-------------

::

    from fivedreg.data import DataHandler
    
    # Initialize handler
    handler = DataHandler()
    
    # Load dataset
    X, y = handler.load_dataset('coursework_dataset.pkl')
    
    # Preprocess and split
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = handler.preprocess_and_split()
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
