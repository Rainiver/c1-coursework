# 5D Neural Network Interpolator

A full-stack system for interpolating 5D datasets, featuring a PyTorch backend and Next.js frontend.

## 🚀 Quick Start (Automated)

The easiest way to run the project is using the provided Docker script:

```bash
# Start everything automatically
./backend/scripts/docker-start.sh

# Stop services
./backend/scripts/docker-stop.sh
```

## 🛠 Manual Setup & Running

If you prefer to run services manually or without Docker, follow these steps.

### 1. Prerequisites
- **Python 3.9+**
- **Node.js 20+**

### 2. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .

# Run the API server (http://localhost:8000)
uvicorn main:app --reload --port 8000
```

### 3. Frontend Setup
```bash
cd frontend
npm install

# Run the Web UI (http://localhost:3000)
npm run dev
```

## 📚 Documentation
For detailed performance analysis and API reference:
```bash
./build_docs.sh  # Builds and opens HTML docs
```

## ⚙️ Environment Config

No .env file is strictly required, but you can configure:

- **HOST/PORT**: Defined in `docker-compose.yml` (default `8000`).
- **MODEL_ARCH**: Hidden layers configured in `backend/interpolator/model.py`.
- **DATA_PATH**: Uploads stored in `backend/uploads/`.

## 📁 Project Structure
```
c1_coursework/
├── backend/          # FastAPI & PyTorch Model
├── frontend/         # Next.js Web Interface
├── docs/             # Sphinx Documentation
└── docker-compose.yml
```
