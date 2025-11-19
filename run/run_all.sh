#!/usr/bin/env bash
################################################################################
# MASTER PIPELINE: Run Complete Causal SLM Evaluation
################################################################################
#
# This script runs the complete experimental pipeline for the Causal SLM project.
# It executes all steps from data generation to figure creation.
#
# Usage:
#   ./run/run_all.sh [--quick]
#
# Options:
#   --quick    Run only essential steps (skip ablations, faster)
#
# Author: Franck Junior Aboya Messou
# Date: October 29, 2025
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR"

# Setup environment
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"

# Parse arguments
QUICK_MODE=false
if [[ "${1:-}" == "--quick" ]]; then
    QUICK_MODE=true
fi

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo ""
    echo -e "${BLUE}========================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================================================${NC}"
    echo ""
}

print_step() {
    echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        print_warning "nvidia-smi not found. GPU may not be available."
        return 1
    fi

    print_step "Checking GPU availability..."
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
}

################################################################################
# Main Pipeline
################################################################################

print_header "CAUSAL SLM - COMPLETE EXPERIMENTAL PIPELINE"

print_step "Project directory: $PROJECT_DIR"
print_step "Quick mode: $QUICK_MODE"

# Check GPU
check_gpu || print_warning "Continuing without GPU check..."

################################################################################
# STEP 1: Generate Synthetic Datasets
################################################################################

print_header "STEP 1: Generate Synthetic Datasets"
print_step "Generating 4 synthetic datasets with ground truth..."

if [[ -f "run/01_generate_datasets.sh" ]]; then
    bash run/01_generate_datasets.sh
else
    python scripts/generate_synthetic_datasets.py
fi

print_step "✓ Datasets generated"

################################################################################
# STEP 2: Run Comprehensive Evaluation
################################################################################

print_header "STEP 2: Run Comprehensive Evaluation"
print_step "Testing all configurations with improved prompts..."

if [[ -f "run/02_run_evaluation.sh" ]]; then
    bash run/02_run_evaluation.sh
else
    timeout 1800 python scripts/run_comprehensive_evaluation.py 2>&1 | \
        tee output/evaluations/comprehensive_evaluation.log
fi

print_step "✓ Comprehensive evaluation complete"

################################################################################
# STEP 3: SOTA Comparisons & Ablations
################################################################################

print_header "STEP 3: SOTA Comparisons & Ablations"
print_step "Comparing against SOTA methods (PCMCI, Correlation)..."

if [[ -f "run/03_sota_comparisons.sh" ]]; then
    bash run/03_sota_comparisons.sh
else
    timeout 1800 python scripts/sota_comparisons_ablations.py 2>&1 | \
        tee output/evaluations/sota_ablations.log
fi

print_step "✓ SOTA comparisons complete"

################################################################################
# STEP 4: Comprehensive Ablations (Optional in quick mode)
################################################################################

if [[ "$QUICK_MODE" == false ]]; then
    print_header "STEP 4: Comprehensive Ablation Studies"
    print_step "Running full Theory×LLM cross-testing (66 tests)..."
    print_warning "This may take 2-3 hours. Use --quick to skip."

    if [[ -f "run/04_comprehensive_ablations.sh" ]]; then
        bash run/04_comprehensive_ablations.sh
    else
        timeout 3600 python scripts/comprehensive_ablations.py 2>&1 | \
            tee output/evaluations/comprehensive_ablations.log
    fi

    print_step "✓ Comprehensive ablations complete"
else
    print_warning "Skipping comprehensive ablations (quick mode)"
fi

################################################################################
# STEP 5: Generate All Figures
################################################################################

print_header "STEP 5: Generate All Publication Figures"
print_step "Creating 12 publication-ready figures (PDF + PNG)..."

if [[ -f "run/05_generate_figures.sh" ]]; then
    bash run/05_generate_figures.sh
else
    python scripts/generate_comprehensive_plots.py
fi

print_step "✓ All figures generated"

################################################################################
# STEP 6: Generate Summary Report
################################################################################

print_header "STEP 6: Generate Summary Report"
print_step "Creating comprehensive summary..."

if [[ -f "run/06_generate_summary.sh" ]]; then
    bash run/06_generate_summary.sh
else
    python scripts/generate_summary_report.py 2>/dev/null || \
        print_warning "Summary report generation not available (optional)"
fi

################################################################################
# Final Status
################################################################################

print_header "PIPELINE COMPLETE!"

echo ""
echo "Results:"
echo "  - Evaluations: output/evaluations/"
echo "  - Figures: output/figures/ (12 PDFs + 12 PNGs)"
echo "  - Narratives: output/narratives/"
echo ""

if [[ "$QUICK_MODE" == true ]]; then
    echo "Note: Ran in quick mode. For full results, run without --quick"
    echo ""
fi

echo "Next steps:"
echo "  1. Review figures: ls -lh output/figures/"
echo "  2. Check results: cat QUICK_RESULTS_REFERENCE.txt"
echo "  3. Read summary: cat FINAL_COMPLETE_SUMMARY.md"
echo ""

print_step "✓ All done! Results ready for paper."

exit 0
