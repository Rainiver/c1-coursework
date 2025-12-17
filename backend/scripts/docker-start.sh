#!/bin/bash
# Start the 5D Interpolator application using Docker Compose
# Supports both 'docker compose' (v2) and 'docker-compose' (v1)

set -e  # Exit on error

# Color codes (define first)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Color output functions
print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}


# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker compose is available (try v2 first, then v1)
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker-compose"
else
    print_error "Docker Compose is not available. Please install Docker Desktop or docker-compose and try again."
    exit 1
fi

print_status "Starting 5D Interpolator in \$MODE mode..."

# Clean up any existing containers
print_status "Cleaning up existing containers..."
$DOCKER_COMPOSE down --remove-orphans

# Build and start services based on mode
if [ "$MODE" = "build" ]; then
    print_status "Building and starting services (this may take a few minutes)..."
    $DOCKER_COMPOSE up --build -d
else
    print_status "Starting services..."
    $DOCKER_COMPOSE up -d
fi

# Wait for services to be ready
print_status "Waiting for services to start..."
sleep 3

# Check if services are running
if docker ps | grep -q "interpolator-backend"; then
    print_success "Backend is running on http://localhost:8000"
else
    print_error "Backend failed to start. Check logs: docker logs interpolator-backend"
    exit 1
fi

if docker ps | grep -q "interpolator-frontend"; then
    print_success "Frontend is running on http://localhost:3000"
else
    print_warning "Frontend may not be running. Check logs: docker logs interpolator-frontend"
fi

echo ""
print_success "5D Interpolator is ready!"
echo ""
echo "  📊 Web App: http://localhost:3000"
echo "  🔧 API Docs: http://localhost:8000/docs"
echo "  ❤️  Health: http://localhost:8000/health"
echo ""
print_status "To stop: ./scripts/docker-stop.sh"
print_status "View logs: docker logs -f interpolator-backend"
