# 5D Neural Network Interpolator

Tiny 5D neural-net interpolator (PyTorch) with normalization & early stopping.

- **Web App**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs

```

### Option 2: Manual Installation

```bash
# Clone/navigate to project
cd interpolator

# Backend setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
cd backend && pip install -e .

# Frontend setup
cd ../frontend && npm install
```

## Quick Start

### Docker (Recommended)

```bash
# Start the full application
./scripts/docker-start.sh

# Visit the web interface
open http://localhost:3000
```

# Stop the application
./scripts/docker-stop.sh
```

### Manual

```bash
# Terminal 1: Backend
cd backend
source venv/bin/activate
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```

## Usage

### Web Interface
1. Upload your 5D dataset (.pkl file)
2. Train the neural network
3. Make predictions with interactive sliders

### CLI Tool

```bash
# Generate synthetic test data
cd backend
python interpolator.py --mode generate --n 1000 --workdir ./test_data

# Quick model test
python interpolator.py --mode test
```

## Project Structure

```
.
├── backend/
│   ├── api/
│   │   └── main.py          # FastAPI endpoints
│   ├── scripts/
│   │   ├── start_server.sh
│   │   └── docker-start.sh
│   ├── data.py              # Data handling
│   ├── interpolator.py      # Core ML module
│   └── Dockerfile
├── frontend/
│   ├── app/
│   │   └── page.tsx         # Main UI
│   └── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## Development

### Testing
```bash
# Run backend tests
cd backend
pytest

# Test pipeline
./scripts/test-pipeline.sh
```

### Documentation
```bash
cd docs
make html
```

## License

MIT
