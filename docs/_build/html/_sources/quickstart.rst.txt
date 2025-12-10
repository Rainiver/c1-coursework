Quick Start Guide
=================

This guide will get you up and running with the Fivedreg Interpolator in 5 minutes.

Step 1: Launch the Application
-------------------------------

::

    ./run_local.sh

Wait for both frontend and backend to start. You should see::

    Backend: http://localhost:8000/docs
    Frontend: http://localhost:3000

Step 2: Upload Dataset
-----------------------

1. Open http://localhost:3000 in your browser
2. Click on **Upload** in the navigation
3. Click **Choose File** and select ``coursework_dataset.pkl``
4. Click **Upload**

You should see dataset statistics including:
- Total samples (5,000)
- Number of features (5)
- Missing values check
- Target distribution (min, max, mean)
- Planned train/val/test split

Step 3: Train the Model
------------------------

1. Navigate to the **Train** page
2. Configure hyperparameters (or use defaults):
   
   - Hidden Layers: ``64,32,16``
   - Learning Rate: ``0.001``
   - Max Epochs: ``100``

3. Click **Start Training**
4. Wait for training to complete (~10-30 seconds for 5K samples)

Training results will display:
- Training time
- Final loss
- Validation MSE
- Validation R² score

Step 4: Make Predictions
-------------------------

1. Navigate to the **Predict** page
2. Enter 5 feature values (e.g., ``0.5, 0.3, 0.7, 0.2, 0.9``)
3. Click **Predict**
4. See the predicted target value

Example Workflow
----------------

Complete workflow example::

    # 1. Start services
    ./run_local.sh
    
    # 2. In browser: Upload coursework_dataset.pkl
    # 3. In browser: Train with default settings
    # 4. In browser: Predict with inputs [0.5, 0.5, 0.5, 0.5, 0.5]

Next Steps
----------

- Read the :doc:`usage` guide for advanced features
- Explore the :doc:`api/index` reference for programmatic access
- Check :doc:`performance` for benchmarking results
