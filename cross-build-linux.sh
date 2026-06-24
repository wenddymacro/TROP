#!/bin/bash
# Cross-compile TROP plugin for Linux x64 using Docker
# Requires: Docker installed and running
#
# This script builds a Linux x64 Stata plugin from macOS/Windows
# using a Docker container with Ubuntu 22.04 + GCC + OpenBLAS.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Building TROP Linux x64 Plugin ==="
echo "Using Docker with Ubuntu 22.04 + GCC + OpenBLAS"
echo ""

# Check Docker availability
if ! command -v docker &>/dev/null; then
    echo "ERROR: Docker is not installed or not in PATH."
    echo "Install Docker Desktop from https://www.docker.com/products/docker-desktop"
    exit 1
fi

if ! docker info &>/dev/null 2>&1; then
    echo "ERROR: Docker daemon is not running."
    echo "Please start Docker Desktop and try again."
    exit 1
fi

# Build Docker image (uses Dockerfile.linux-build)
echo "--- Step 1: Building Docker image ---"
docker build -f Dockerfile.linux-build -t trop-linux-builder .

# Ensure output directory exists
mkdir -p "$SCRIPT_DIR/ado"

# Run container and extract plugin to ado/
echo ""
echo "--- Step 2: Extracting plugin ---"
docker run --rm -v "$SCRIPT_DIR/ado:/output" trop-linux-builder

# Verify output
echo ""
if [ -f "$SCRIPT_DIR/ado/trop_linux_x64.plugin" ]; then
    echo "=== SUCCESS ==="
    file "$SCRIPT_DIR/ado/trop_linux_x64.plugin"
    ls -lh "$SCRIPT_DIR/ado/trop_linux_x64.plugin"
else
    echo "=== FAILED: trop_linux_x64.plugin not produced ==="
    exit 1
fi
