# 5D Neural Network Interpolator

A full-stack neural network system for interpolating 5-dimensional numerical datasets, built with PyTorch, FastAPI, and Next.js.

## 🔗 Repository

**GitLab:** [https://github.com/Rainiver/c1-coursework](https://github.com/Rainiver/c1-coursework)

## ✨ Features

- **PyTorch Neural Network:** Configurable MLP with [256, 128, 64, 32] architecture
- **FastAPI Backend:** REST API for upload, training, and prediction
- **Next.js Frontend:** Modern, responsive web interface
- **Docker Deployment:** One-command containerized deployment
- **High Performance:** RMSE 0.0246, R² 0.9946 on test data
- **Fast Training:** < 1 second for 5000 samples

## 🚀 Quick Start

### Docker (Recommended)

```bash
# Start both backend and frontend
./backend/scripts/docker-start.sh

# Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs

# Stop services
./backend/scripts/docker-stop.sh
```

### Local Development

#### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend

```bash
cd frontend
npm install
npm run dev
# Access at http://localhost:3000
```

## 📖 Usage

### Web Interface

1. **Upload Dataset:** Click "Upload Dataset" and select your `.pkl` file
2. **Train Model:** Configure epochs (default: 200) and click "Start Training"
3. **Make Predictions:** Use the 5 input sliders and click "Predict"

### API

```bash
# Upload dataset
curl -X POST http://localhost:8000/upload \
  -F "file=@coursework_dataset.pkl"

# Train model
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"hidden_layers": [256,128,64,32], "learning_rate": 0.005, "max_epochs": 200}'

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": [1.0, 2.0, 3.0, 4.0, 5.0]}'
```

### CLI

```bash
cd backend

# Test with synthetic data
python interpolator.py

# Generate custom dataset
python interpolator.py --mode generate --n 1000 --workdir ./test_data
```

## 📁 Project Structure

```
c1_coursework/
├── backend/
│   ├── api/
│   │   └── main.py              # FastAPI endpoints
│   ├── scripts/
│   │   ├── docker-start.sh      # Launch Docker stack
│   │   ├── docker-stop.sh       # Stop Docker stack
│   │   ├── docker-logs.sh       # View logs
│   │   ├── benchmark_performance.py
│   │   └── tune_hyperparameters.py
│   ├── tests/                   # Test suite
│   ├── data.py                  # Data handling module
│   ├── interpolator.py          # Core ML module
│   ├── pyproject.toml
│   └── Dockerfile
├── frontend/
│   ├── app/
│   │   ├── page.tsx             # Main UI
│   │   └── globals.css
│   ├── package.json
│   └── Dockerfile
├── docs/                        # Sphinx documentation
│   ├── conf.py
│   ├── index.rst
│   ├── installation.rst
│   ├── quickstart.rst
│   ├── usage.rst
│   ├── performance.rst
│   ├── testing.rst
│   └── api/
├── docker-compose.yml
├── build_docs.sh                # Build Sphinx docs
├── README.md
└── pyproject.toml
```

## 🧪 Testing

```bash
cd backend

# Run all tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Specific test file
pytest tests/test_model.py
```

## 📊 Performance Benchmarking

```bash
cd backend
python scripts/benchmark_performance.py
```

Tests dataset sizes: 500, 1K, 2K, 3.5K, 5K, 10K samples.  
Results saved to: `performance_benchmark_results.json`

## 📚 Documentation

Build Sphinx documentation:

```bash
./build_docs.sh
```

Documentation will open automatically in your browser, or access at:
`file://path/to/docs/_build/html/index.html`

## 🔧 Development

### Running Tests

```bash
cd backend
pytest -v
```

### Performance Tuning

```bash
cd backend
python scripts/tune_hyperparameters.py
```

### Docker

```bash
# Rebuild containers
docker-compose up -d --build

# View logs
./backend/scripts/docker-logs.sh

# Restart services
docker-compose restart
```

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| Test RMSE | 0.024635 |
| Test R² | 0.9946 |
| Training Time (5K samples) | 0.47s |
| Architecture | [256, 128, 64, 32] |

## 🛠️ Technology Stack

**Backend:**
- Python 3.9+
- PyTorch (neural network)
- FastAPI (REST API)
- Uvicorn (ASGI server)

**Frontend:**
- Next.js 15
- React 19
- TypeScript
- Tailwind CSS

**Deployment:**
- Docker
- Docker Compose

**Documentation:**
- Sphinx

## 📝 Requirements

- Python 3.9+
- Node.js 20+
- Docker & Docker Compose (for containerized deployment)
- 8GB+ RAM recommended

## 📄 License

MIT License - See LICENSE file for details

## 👤 Author

C1 Research Computing Coursework 2025  
MPhil in Data Intensive Science  
University of Cambridge

---

**Documentation:** Run `./build_docs.sh` for complete API reference and guides  
**Support:** See [docs/](docs/) for detailed usage instructions
