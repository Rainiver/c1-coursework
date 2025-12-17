#!/bin/bash
# Stop the 5D Interpolator Docker containers

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if docker compose is available (try v2 first, then v1)
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker-compose"
else
    print_error "Docker Compose is not available."
    exit 1
fi

print_status "Stopping 5D Interpolator containers..."

# Stop and remove containers
$DOCKER_COMPOSE down

# Optional: remove volumes (commented out by default)
# $DOCKER_COMPOSE down -v

print_success "Containers stopped successfully!"
