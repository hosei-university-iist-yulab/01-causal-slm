#!/usr/bin/env bash
###############################################################################
# Generate All Publication-Ready Figures
###############################################################################
#
# Generates 12 figures (PDF + PNG, 300 DPI):
#   - Figures 1-4:  Main results (original)
#   - Figures 5-12: Comprehensive analysis (new)
#
# Usage: ./run/05_generate_figures.sh
###############################################################################

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="src:$PYTHONPATH"

echo "==> Generating all publication figures..."
python scripts/generate_comprehensive_plots.py

echo ""
echo "✓ Figures generated in output/figures/"
echo ""
echo "Generated files:"
ls -lh output/figures/*.pdf 2>/dev/null | awk '{print $9, $5}'
echo ""
echo "Total: 12 PDFs + 12 PNGs"
