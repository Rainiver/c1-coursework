Installation Guide
==================

Prerequisites
-------------

- Python 3.9 or higher
- Node.js 18+ and npm
- Git

Quick Installation
-------------------

1. **Clone the repository**::

    git clone https://github.com/Rainiver/c1-coursework.git
    cd c1-coursework

2. **Install backend dependencies**::

    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\\Scripts\\activate
    pip install -e ./backend

3. **Install frontend dependencies**::

    cd frontend
    npm install
    cd ..

4. **Run the application**::

    ./run_local.sh

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

Docker Installation (Alternative)
----------------------------------

If you prefer using Docker::

    docker-compose up --build

This will start both frontend and backend services in containers.

Verifying Installation
-----------------------

1. Open http://localhost:3000 in your browser
2. Navigate to the Upload page
3. Upload the provided ``coursework_dataset.pkl`` file
4. Verify that dataset statistics are displayed
5. Go to Train page and start training
6. Check that training completes successfully with metrics displayed

Troubleshooting
---------------

**Backend won't start:**
- Ensure Python 3.9+ is installed
- Check that all dependencies are installed: ``pip list``
- Verify virtual environment is activated

**Frontend won't start:**
- Ensure Node.js 18+ is installed: ``node --version``
- Delete ``node_modules`` and ``package-lock.json``, then run ``npm install`` again
- Check that port 3000 is not already in use

**API calls failing:**
- Verify backend is running on port 8000
- Check ``next.config.ts`` has correct proxy configuration
- Look for CORS errors in browser console
