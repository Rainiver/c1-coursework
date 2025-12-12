Installation
============

This guide covers installation for both local development and Docker deployment.

Prerequisites
-------------

Docker Deployment (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Docker (>= 20.10)
* Docker Compose (>= 2.0)
* 8GB+ RAM

Local Development
~~~~~~~~~~~~~~~~~

* Python 3.9+
* Node.js 20+
* 8GB+ RAM
* macOS, Linux, or Windows with WSL2

Docker Deployment
-----------------

1. **Clone the repository**::

    git clone https://github.com/Rainiver/c1-coursework.git
    cd c1_coursework

2. **Start services**::

    ./backend/scripts/docker-start.sh

3. **Access application**:

   * Frontend: http://localhost:3000
   * Backend API: http://localhost:8000
   * API Docs: http://localhost:8000/docs

4. **Stop services**::

    ./backend/scripts/docker-stop.sh

Local Development
-----------------

Backend Setup
~~~~~~~~~~~~~

1. **Create virtual environment**::

    cd backend
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\\Scripts\\activate

2. **Install dependencies**::

    pip install -e .

3. **Start backend server**::

    ./scripts/start_server.sh
    # Or manually: uvicorn api.main:app --reload

Frontend Setup
~~~~~~~~~~~~~~

1. **Install dependencies**::

    cd frontend
    npm install

2. **Start development server**::

    npm run dev

3. **Access at**: http://localhost:3000

Verification
------------

Test that everything works::

    # Check backend health
    curl http://localhost:8000/health

    # Run backend tests
    cd backend
    pytest

Common Issues
-------------

**Port already in use**
  Stop existing services on ports 3000 or 8000

**Docker out of memory**
  Increase Docker memory limit to 4GB+

**Module not found errors**
  Ensure virtual environment is activated
