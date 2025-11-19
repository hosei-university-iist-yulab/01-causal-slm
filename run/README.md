# Run Scripts - Unified Execution Pipeline

This directory contains unified bash scripts to execute the complete Causal SLM experimental pipeline.

## Quick Start

```bash
# Run complete pipeline (all steps)
./run/run_all.sh

# Run quick version (skip long ablations)
./run/run_all.sh --quick

# Or run individual steps:
./run/01_generate_datasets.sh
./run/02_run_evaluation.sh
./run/03_sota_comparisons.sh
./run/04_comprehensive_ablations.sh  # Optional, takes 2-3 hours
./run/05_generate_figures.sh
```

## Available Scripts

### Master Pipeline

- **`run_all.sh`** - Run complete experimental pipeline
  - Generates datasets
  - Runs evaluations
  - Performs SOTA comparisons
  - Runs ablation studies (optional with --quick)
  - Generates all figures
  - Creates summary report
  - **Usage:** `./run/run_all.sh [--quick]`
  - **Time:** ~30 min (quick mode) or ~3 hours (full)

### Individual Steps

1. **`01_generate_datasets.sh`** - Generate 4 synthetic datasets with ground truth
   - Output: `data/*.npz`, `data/*.json`
   - Time: ~2 min

2. **`02_run_evaluation.sh`** - Run comprehensive evaluation with improved prompts
   - Output: `output/evaluations/comprehensive_evaluation_*.json`
   - Time: ~10-15 min

3. **`03_sota_comparisons.sh`** - SOTA method comparison (PCMCI vs Correlation)
   - Output: `output/evaluations/sota_ablations_*.json`
   - Time: ~5-10 min

4. **`04_comprehensive_ablations.sh`** - Full Theory×LLM ablation matrix (66 tests)
   - Output: `output/evaluations/comprehensive_ablations_*.json`
   - Time: ~2-3 hours (can skip with run_all.sh --quick)

5. **`05_generate_figures.sh`** - Generate all 12 publication figures
   - Output: `output/figures/*.pdf`, `output/figures/*.png`
   - Time: ~5 seconds

6. **`06_generate_summary.sh`** - Create comprehensive summary report
   - Output: Console summary + optional report files
   - Time: ~1 second

### Setup

- **`setup_environment.sh`** - Setup Python environment and dependencies
  - Installs required packages
  - Checks GPU availability
  - Verifies configuration

## Requirements

### Hardware
- **GPUs:** 3× NVIDIA GPUs (RTX 3090 or similar, 24GB+ VRAM each)
- **GPUs Used:** 5, 6, 7 (for GPT-2, TinyLlama, Phi-2)
- **RAM:** 32GB+ recommended
- **Disk:** 5GB+ free space

### Software
- Python 3.8+
- CUDA 11.0+ (for GPU support)
- See `requirements.txt` for Python packages

## Project Structure

```
01-causal-slm/
├── run/                    # ← You are here
│   ├── run_all.sh         # Master pipeline
│   ├── 01_generate_datasets.sh
│   ├── 02_run_evaluation.sh
│   ├── 03_sota_comparisons.sh
│   ├── 04_comprehensive_ablations.sh
│   ├── 05_generate_figures.sh
│   ├── 06_generate_summary.sh
│   ├── setup_environment.sh
│   └── README.md          # This file
│
├── scripts/               # Python scripts (called by run/)
│   ├── generate_synthetic_datasets.py
│   ├── run_comprehensive_evaluation.py
│   ├── sota_comparisons_ablations.py
│   ├── comprehensive_ablations.py
│   └── generate_comprehensive_plots.py
│
├── src/                   # Source code (core library)
│   ├── causal_discovery/  # Causal discovery methods
│   ├── llm_integration/   # Multi-LLM system
│   ├── data_generation/   # Dataset generators
│   └── utils/             # Utilities
│
└── output/                # Results (generated)
    ├── evaluations/       # JSON results
    ├── figures/           # Publication figures (PDF+PNG)
    └── narratives/        # Generated narratives
```

## Example Workflows

### For Beginners (First Time Users)

```bash
# 1. Setup environment
cd /path/to/01-causal-slm
./run/setup_environment.sh

# 2. Run quick pipeline (skip long ablations)
./run/run_all.sh --quick

# 3. Check results
ls -lh output/figures/
cat QUICK_RESULTS_REFERENCE.txt
```

**Time:** ~30 minutes

### For Expert Users (Full Reproduction)

```bash
# Run complete pipeline with all ablations
./run/run_all.sh

# Or run steps individually for fine control:
./run/01_generate_datasets.sh
./run/02_run_evaluation.sh
./run/03_sota_comparisons.sh
./run/04_comprehensive_ablations.sh  # Long! 2-3 hours
./run/05_generate_figures.sh
```

**Time:** ~3 hours total

### For Paper Submission (Just Figures)

If you already have results and just need to regenerate figures:

```bash
./run/05_generate_figures.sh
```

**Time:** ~5 seconds

### For Development (Single Component)

```bash
# Test only causal discovery
cd /path/to/01-causal-slm
export PYTHONPATH=src:$PYTHONPATH
python scripts/sota_comparisons_ablations.py

# Test only LLM narratives
python scripts/run_comprehensive_evaluation.py

# Test only figure generation
python scripts/generate_comprehensive_plots.py
```

## Troubleshooting

### GPU Issues

```bash
# Check GPUs
nvidia-smi

# Check GPU availability in Python
python -c "import torch; print(torch.cuda.is_available())"

# Specify different GPUs (edit scripts or set env)
export CUDA_VISIBLE_DEVICES=0,1,2
```

### Python Path Issues

```bash
# Ensure PYTHONPATH is set
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# Or use absolute path
export PYTHONPATH="/full/path/to/01-causal-slm/src:$PYTHONPATH"
```

### Memory Issues

```bash
# Reduce batch size in scripts (edit Python files)
# Or run with fewer GPUs
# Or run steps individually (don't run run_all.sh)
```

### Permission Issues

```bash
# Make scripts executable
chmod +x run/*.sh
```

## Output Files

After running the pipeline, you'll have:

### Evaluation Results
- `output/evaluations/comprehensive_evaluation_*.json` - Main results
- `output/evaluations/sota_ablations_*.json` - SOTA comparison
- `output/evaluations/comprehensive_ablations_*.json` - Full ablations

### Figures (Publication-Ready)
- `output/figures/figure1_causal_discovery_sota.pdf` - SOTA comparison
- `output/figures/figure2_llm_ablation.pdf` - LLM ablation
- `output/figures/figure3_improved_prompts.pdf` - Prompt improvements
- `output/figures/figure4_sota_detailed_metrics.pdf` - Detailed metrics
- `output/figures/figure5_radar_performance.pdf` - Radar chart
- `output/figures/figure6_complexity_line_plot.pdf` - Complexity analysis
- `output/figures/figure7_f1_distributions.pdf` - Statistical distributions
- `output/figures/figure8_performance_heatmap.pdf` - Performance matrix
- `output/figures/figure9_stacked_metrics.pdf` - Metric breakdown
- `output/figures/figure10_precision_recall_scatter.pdf` - PR trade-off
- `output/figures/figure11_violin_llm_lengths.pdf` - LLM distributions
- `output/figures/figure12_simple_comparison.pdf` - Simple overview

(+ PNG versions of all figures)

### Documentation
- `FINAL_COMPLETE_SUMMARY.md` - Complete project summary
- `QUICK_RESULTS_REFERENCE.txt` - Quick results reference
- `FIGURES_CATALOG.md` - Complete figure documentation

## Performance Metrics

All scripts track and report:
- **Runtime** (wall time, CPU time)
- **Memory usage** (RAM + GPU)
- **CO2 footprint** (via codecarbon)

Results saved in evaluation JSON files.

## Citation

If you use this code, please cite:

```bibtex
@article{aboya2025causal,
  title={Multi-LLM Ensemble for Causal Discovery and Narrative Generation},
  author={Aboya Messou, Franck Junior and others},
  journal={TBD},
  year={2025}
}
```

## Support

For issues or questions:
1. Check this README
2. Review `FINAL_COMPLETE_SUMMARY.md`
3. Check individual script comments
4. Open an issue on GitHub (TBD)

## License

[TBD]

---

**Last Updated:** October 29, 2025
**Author:** Franck Junior Aboya Messou
**Status:** Production-ready, tested, publication-quality
