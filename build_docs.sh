#!/bin/bash
# Build Sphinx documentation

set -e

echo "======================================"
echo "Building Sphinx Documentation"
echo "======================================"

# Install Sphinx if needed
echo "Ensuring Sphinx is installed..."
python3 -m pip install --quiet sphinx

# Build HTML documentation
echo "Building HTML docs..."
cd docs
python3 -m sphinx -b html . _build/html

echo ""
echo "======================================"
echo "✓ Documentation built successfully!"
echo "======================================"
echo ""
echo "View documentation at:"
echo "  file://$(pwd)/_build/html/index.html"
echo ""
echo "Or open in browser:"
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "  open _build/html/index.html"
    # Auto-open on macOS
    open _build/html/index.html 2>/dev/null || true
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "  xdg-open _build/html/index.html"
else
    echo "  (Open _build/html/index.html in your browser)"
fi

