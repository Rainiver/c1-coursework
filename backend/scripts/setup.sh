#!/bin/bash
cd "$(dirname "$0")"
source ../venv/bin/activate  # On Windows: venv\Scripts\activate
cd backend && pip install -e .
