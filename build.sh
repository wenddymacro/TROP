#!/bin/bash
# Build script for TROP Stata plugin
# Builds: Rust core library, C plugin, and optionally Mata library (ltrop.mlib)
#
# Usage:
#   ./build.sh          - Build native plugin (macOS/Linux)
#   ./build.sh --linux  - Cross-compile Linux x64 plugin via Docker

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --linux flag: cross-compile for Linux x64 using Docker
if [[ "${1:-}" == "--linux" ]]; then
    echo "=== Cross-compiling for Linux x64 (Docker) ==="
    exec bash "$SCRIPT_DIR/cross-build-linux.sh"
fi

echo "=== Building Rust library ==="
cd "$SCRIPT_DIR/rust"
cargo build --release

echo "=== Building C plugin ==="
cd "$SCRIPT_DIR/plugin"
make

echo "=== Plugins built ==="
ls -la "$SCRIPT_DIR/plugin/"*.plugin

# === Build Mata library (optional, requires Stata) ===
# ltrop.mlib packages all Mata functions for distribution.
# When installed via `net install`, Stata auto-loads functions from mlib.
# Skip if Stata is not available (development still works via load_mata_once.do).
echo ""
echo "=== Building Mata library (ltrop.mlib) ==="

# Try to find Stata executable
STATA_EXEC=""
for cmd in stata-mp stata-se stata; do
    if command -v "$cmd" &>/dev/null; then
        STATA_EXEC="$cmd"
        break
    fi
done

if [ -n "$STATA_EXEC" ]; then
    echo "Using Stata: $STATA_EXEC"
    cd "$SCRIPT_DIR"
    # Create temp wrapper that cd's to SCRIPT_DIR before running build_mlib.do.
    # This is needed because profile.do may change c(pwd) at Stata startup.
    TMPDO=$(mktemp /tmp/build_mlib_wrapper.XXXXXX.do)
    echo "cd \"$SCRIPT_DIR\"" > "$TMPDO"
    echo "do \"$SCRIPT_DIR/build_mlib.do\"" >> "$TMPDO"
    "$STATA_EXEC" -b do "$TMPDO"
    rm -f "$TMPDO"
    # Move log from c(pwd) (which profile.do may have changed) to SCRIPT_DIR
    if [ -f "$SCRIPT_DIR/ltrop.mlib" ]; then
        echo "ltrop.mlib created successfully"
    else
        echo "WARNING: ltrop.mlib build failed. Check build_mlib.log for details."
        echo "Package will still work in development mode (source compilation at runtime)."
    fi
else
    echo "WARNING: Stata not found in PATH. Skipping ltrop.mlib build."
    echo "For distribution, run 'stata-mp -b do build_mlib.do' from trop_stata/ directory."
    echo "Development mode still works via load_mata_once.do (source compilation at runtime)."
fi

echo ""
echo "=== Build complete ==="
