#!/bin/bash
set -e

# Create virtualenv if not exists
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

# Install backend dependencies
pip install --upgrade pip
pip install -e ./backend
pip install sphinx

# Build docs
cd docs
make html
echo "Documentation built in docs/_build/html/index.html"
open _build/html/index.html || echo "Open docs/_build/html/index.html in your browser."
