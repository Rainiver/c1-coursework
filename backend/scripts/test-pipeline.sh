#!/bin/bash
# Test pipeline script for the interpolator project

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_status() {
    echo ""
    echo "$1"
}

# Ensure cleanup happens on script exit (success, failure, or interrupt)
trap cleanup_test_env EXIT

# Run the test
print_status "Running pipeline test..."
echo ""

if [ "$MODE" = "quick" ]; then
    python3 "$SCRIPT_PATH" --host "$HOST" --port "$PORT" --quick
else
    python3 "$SCRIPT_PATH" --host "$HOST" --port "$PORT"
fi

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    print_success "Pipeline test completed successfully! 🎉"
else
    print_error "Pipeline test failed! ❌"
    echo ""
    print_status "Troubleshooting tips:"
    echo "  1. Make sure backend is running: ./scripts/docker-start.sh"
    echo "  2. Check backend logs: ./scripts/docker-logs.sh backend"
    echo "  3. Verify API health: curl http://\$HOST:\$PORT/health"
    echo "  4. Try quick test: $0 quick"
fi

exit $exit_code
