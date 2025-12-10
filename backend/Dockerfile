FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY backend/pyproject.toml .
RUN pip install --no-cache-dir .

# Copy backend code
COPY backend/ .

# Expose API port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
