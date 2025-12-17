User Guide
==========

Core Workflow
-----------

1. **Upload Dataset**
   - Click "Upload Dataset" in the web interface.
   - Select your `.pkl` file (must contain 5D input coordinates and target values).

2. **Train Model**
   - Navigate to the "Train" section.
   - (Optional) Adjust hyperparameters:
     - **Hidden Layers:** Structure of the neural network (e.g., ``[256, 128, 64]``).
     - **Learning Rate:** Step size for optimization (default: ``0.001``).
     - **Max Epochs:** Training duration (default: ``200``).
   - Click "Start Training". Progress is shown in real-time.

3. **Make Predictions**
   - Once trained, use the 5 sliders to set input coordinates covering the data range.
   - Click "Predict" to query the model.
   - The result and inference time will be displayed instantly.

Command Line Interface (CLI)
--------------------------

For batch operations or automation, use the CLI tools in the ``backend/`` directory:

.. code-block:: bash

   # Generate synthetic test data
   python interpolator.py --mode generate --n 1000 --workdir ./data

   # Train model from terminal
   python interpolator.py --mode train --file ./data/dataset.pkl

   # Predict a single point
   python interpolator.py --mode predict --input "[0.5, 0.5, 0.5, 0.5, 0.5]"
