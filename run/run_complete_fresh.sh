#!/usr/bin/env bash
################################################################################
# COMPLETE FRESH RUN - From Scratch with CSM 500 Epochs
################################################################################
#
# This script runs the COMPLETE experimental pipeline from scratch:
# 1. Generates datasets
# 2. Runs comprehensive evaluation
# 3. Runs SOTA comparisons
# 4. Runs comprehensive ablations (FULL, not quick)
# 5. Generates all 12 figures
# 6. Commits results
#
# CSM FIXED TO 500 EPOCHS FOR OPTIMAL PERFORMANCE
#
# Usage: ./run/run_complete_fresh.sh
#
# Estimated time: 3-4 hours
################################################################################

set -e

cd "$(dirname "$(dirname "$0")")"

export PYTHONPATH="src:${PYTHONPATH:-}"

echo "====================================================================="
echo "COMPLETE FRESH RUN - ALL EXPERIMENTS FROM SCRATCH"
echo "====================================================================="
echo ""
echo "CSM Configuration: 500 epochs (optimal)"
echo "Expected duration: 3-4 hours"
echo ""

################################################################################
# STEP 1: Generate Datasets
################################################################################
echo ""
echo "====================================================================="
echo "[1/5] GENERATING SYNTHETIC DATASETS"
echo "====================================================================="
python scripts/generate_datasets.py
echo "✓ Datasets generated"

################################################################################
# STEP 2: Comprehensive Evaluation
################################################################################
echo ""
echo "====================================================================="
echo "[2/5] RUNNING COMPREHENSIVE EVALUATION (Improved Prompts)"
echo "====================================================================="
timeout 1800 python scripts/run_comprehensive_evaluation.py 2>&1 | \
    tee output/evaluations/comprehensive_evaluation.log
echo "✓ Comprehensive evaluation complete"

################################################################################
# STEP 3: SOTA Comparisons
################################################################################
echo ""
echo "====================================================================="
echo "[3/5] RUNNING SOTA COMPARISONS (PCMCI vs Correlation)"
echo "====================================================================="
timeout 1800 python scripts/sota_comparisons_ablations.py 2>&1 | \
    tee output/evaluations/sota_comparisons.log
echo "✓ SOTA comparisons complete"

################################################################################
# STEP 4: Comprehensive Ablations (FULL, Theory×LLM matrix)
################################################################################
echo ""
echo "====================================================================="
echo "[4/5] RUNNING COMPREHENSIVE ABLATIONS (Theory×LLM Full Matrix)"
echo "====================================================================="
echo "This will take 2-3 hours for 66 tests across 3 datasets..."
timeout 10800 python scripts/comprehensive_ablations.py 2>&1 | \
    tee output/evaluations/comprehensive_ablations_full.log
echo "✓ Comprehensive ablations complete"

################################################################################
# STEP 5: Generate All Figures
################################################################################
echo ""
echo "====================================================================="
echo "[5/5] GENERATING ALL 12 PUBLICATION FIGURES"
echo "====================================================================="
python scripts/generate_comprehensive_plots.py
echo "✓ All figures generated"

################################################################################
# Summary
################################################################################
echo ""
echo "====================================================================="
echo "COMPLETE PIPELINE FINISHED!"
echo "====================================================================="
echo ""
echo "Results:"
echo "  - Evaluations: output/evaluations/"
ls -lh output/evaluations/*.json 2>/dev/null | awk '{print "    " $9, "(" $5 ")"}'
echo ""
echo "  - Figures: output/figures/"
ls -lh output/figures/*.pdf 2>/dev/null | wc -l | awk '{print "    " $1 " PDF files generated"}'
echo ""
echo "Next: Review results and commit to git"
echo ""

exit 0
