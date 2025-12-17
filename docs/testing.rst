Test Suite
==========

The project includes a comprehensive test suite using ``pytest`` to ensure reliability of the model, API, and data utilities.

Running Tests
-----------

To execute the full test suite:

.. code-block:: bash

   cd backend
   pytest

Specific Tests
------------

You can run specific test categories if needed:

.. code-block:: bash

   # Model architecture and logic tests
   pytest tests/test_model.py

   # API endpoint tests
   pytest tests/test_api.py

   # Data processing tests
   pytest tests/test_data.py
