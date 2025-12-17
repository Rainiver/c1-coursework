Installation
============

Quick Start (Recommended)
-----------------------

The easiest way to run the 5D Interpolator is using the provided Docker script, which handles both backend and frontend automatically.

**Prerequisites:**
- Docker Desktop (installed and running)

.. code-block:: bash

   # Start the application
   ./backend/scripts/docker-start.sh

Access the application at:
- **Web Interface:** http://localhost:3000
- **API Documentation:** http://localhost:8000/docs

Local Development Setup
---------------------

If you prefer to run services locally without Docker:

**1. Backend Setup**

.. code-block:: bash

   cd backend
   python -m venv venv
   source venv/bin/activate
   pip install -e .
   uvicorn api.main:app --reload --port 8000

**2. Frontend Setup**

.. code-block:: bash

   cd frontend
   npm install
   npm run dev

The application will be available at the same URLs as above.
