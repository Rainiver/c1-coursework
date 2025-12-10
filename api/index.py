import sys
import os

# Add backend to path so we can import 'fivedreg' and 'main'
sys.path.append(os.path.join(os.path.dirname(__file__), '../backend'))

from main import app

# Vercel expects 'app' to be available for ASGI
