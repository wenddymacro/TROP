#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/rust"

echo "=== Running TROP Performance Benchmarks ==="
echo "Working directory: $(pwd)"
echo ""

cargo bench --bench distance_bench "$@"

echo ""
echo "=== Benchmarks Complete ==="
echo "HTML reports: target/criterion/report/index.html"
