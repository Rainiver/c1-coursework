FROM python:3.9-slim

WORKDIR /app

# Copy only requirements first for better caching
COPY pyproject.toml ./

# Install dependencies
RUN pip install --no-cache-dir numpy torch fastapi uvicorn python-multipart pandas scipy psutil requests scikit-learn

# Copy backend code
COPY . .

# Create necessary directories
RUN mkdir -p uploads models data

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "-m", "uvicorn", "api.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]

