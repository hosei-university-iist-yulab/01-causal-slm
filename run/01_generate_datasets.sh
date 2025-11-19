#!/usr/bin/env bash
###############################################################################
# Generate Synthetic Datasets with Ground Truth
###############################################################################
#
# Generates 4 synthetic datasets:
#   1. HVAC Simple (4 variables)
#   2. Industrial Simple (4 variables)
#   3. HVAC Complex (6 variables)
#   4. Environmental Complex (7 variables)
#
# Usage: ./run/01_generate_datasets.sh
###############################################################################

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="src:$PYTHONPATH"

echo "==> Generating synthetic datasets..."
python scripts/generate_synthetic_datasets.py

echo "✓ Datasets generated in data/"
echo ""
echo "Generated files:"
ls -lh data/*.npz data/*.json 2>/dev/null || echo "No files found"
