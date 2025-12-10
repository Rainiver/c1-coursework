Model Module
============

.. automodule:: fivedreg.model
   :members:
   :undoc-members:
   :show-inheritance:

ModelHandler Class
------------------

.. autoclass:: fivedreg.model.ModelHandler
   :members:
   :undoc-members:
   :show-inheritance:
   
   .. automethod:: __init__
   .. automethod:: fit
   .. automethod:: predict

Usage Example
-------------

::

    from fivedreg.model import ModelHandler
    import numpy as np
    
    # Initialize model with custom config
    model = ModelHandler(
        hidden_layers=[128, 64, 32],
        learning_rate=0.0005,
        max_epochs=150
    )
    
    # Train
    X_train = np.random.rand(1000, 5)
    y_train = np.random.rand(1000)
    
    results = model.fit(X_train, y_train)
    print(f"Training time: {results['training_time']:.4f}s")
    print(f"Final loss: {results['losses'][-1]:.6f}")
    
    # Predict
    X_new = np.random.rand(10, 5)
    predictions = model.predict(X_new)
    print(f"Predictions: {predictions}")
