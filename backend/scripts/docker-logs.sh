#!/bin/bash
# View logs from Docker containers

CONTAINER=${1:-backend}

if [ "$CONTAINER" = "backend" ] || [ "$CONTAINER" = "b" ]; then
    docker logs -f interpolator-backend
elif [ "$CONTAINER" = "frontend" ] || [ "$CONTAINER" = "f" ]; then
    docker logs -f interpolator-frontend
else
    echo "Usage: ./docker-logs.sh [backend|frontend]"
    echo "Default: backend"
    exit 1
fi
