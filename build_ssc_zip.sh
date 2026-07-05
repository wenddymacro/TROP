#!/bin/bash
# build_ssc_zip.sh — Generate flat ZIP for SSC submission
# Usage: cd trop_stata && ./build_ssc_zip.sh
#
# Output: /tmp/trop_ssc/trop.zip (flat, no subdirectories)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="/tmp/trop_ssc"
ZIP_NAME="trop.zip"

echo "=== TROP SSC Package Builder ==="
echo "Source: $SCRIPT_DIR"
echo "Output: $OUT_DIR/$ZIP_NAME"
echo ""

# Clean previous build
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

# --- Copy distribution files (flat) ---

# ADO files, help files, plugins
cp "$SCRIPT_DIR"/ado/*.ado "$OUT_DIR/" 2>/dev/null || true
cp "$SCRIPT_DIR"/ado/*.sthlp "$OUT_DIR/" 2>/dev/null || true
cp "$SCRIPT_DIR"/ado/*.plugin "$OUT_DIR/" 2>/dev/null || true

# Precompiled Mata library (CRITICAL for SSC)
cp "$SCRIPT_DIR/ltrop.mlib" "$OUT_DIR/"

# Mata source files (fallback compilation support)
cp "$SCRIPT_DIR"/mata/*.mata "$OUT_DIR/" 2>/dev/null || true
cp "$SCRIPT_DIR/mata/compile_all.do" "$OUT_DIR/" 2>/dev/null || true

# Load helper
cp "$SCRIPT_DIR/load_mata_once.do" "$OUT_DIR/"

# License
cp "$SCRIPT_DIR/LICENSE" "$OUT_DIR/"

# stata.toc
cp "$SCRIPT_DIR/stata.toc" "$OUT_DIR/"

# --- Generate flat trop.pkg ---
# Transform: remove all directory prefixes from f/g lines
sed -E 's|^f ado/|f |; s|^f mata/|f |; s|^g ancillary/|g |' "$SCRIPT_DIR/trop.pkg" > "$OUT_DIR/trop.pkg"

# --- Create ZIP (flat, no paths) ---
cd "$OUT_DIR"
rm -f "$ZIP_NAME"
zip -j "$ZIP_NAME" * 2>/dev/null

# --- Summary ---
echo ""
echo "=== Build Complete ==="
echo "ZIP: $OUT_DIR/$ZIP_NAME"
echo ""
echo "File count: $(unzip -l "$ZIP_NAME" | grep -c "^\s")"
echo "ZIP size: $(du -h "$ZIP_NAME" | cut -f1)"
echo ""
echo "=== Contents ==="
unzip -l "$ZIP_NAME" | grep -v "^Archive\|^---\|^\s*$\|files$"
echo ""

# --- Verification ---
echo "=== Verification ==="
ERRORS=0

# Check no subdirectories
if unzip -l "$ZIP_NAME" | grep -q "/.*[^/]"; then
    # This is fine - files without dirs
    true
fi
SUBDIRS=$(unzip -l "$ZIP_NAME" | awk '{print $NF}' | grep "/" | grep -v "^$" || true)
if [ -n "$SUBDIRS" ]; then
    echo "WARNING: Found paths with directories:"
    echo "$SUBDIRS"
    ERRORS=$((ERRORS + 1))
fi

# Check critical files
for f in trop.pkg trop.ado trop.sthlp trop_estat.sthlp trop_predict.sthlp \
         ltrop.mlib load_mata_once.do \
         trop_macos_arm64.plugin trop_macos_x64.plugin \
         trop_linux_x64.plugin trop_windows_x64.plugin; do
    if unzip -l "$ZIP_NAME" | grep -q " $f$"; then
        echo "  ✓ $f"
    else
        echo "  ✗ MISSING: $f"
        ERRORS=$((ERRORS + 1))
    fi
done

if [ $ERRORS -eq 0 ]; then
    echo ""
    echo "=== ALL CHECKS PASSED ==="
    echo "Ready for SSC submission: $OUT_DIR/$ZIP_NAME"
else
    echo ""
    echo "=== $ERRORS ERROR(S) FOUND ==="
    exit 1
fi
