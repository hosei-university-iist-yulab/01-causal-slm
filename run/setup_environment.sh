#!/bin/bash

# Setup script for Causal SLM environment
# Topic 1: Causal Language Models for Temporal Sensor Sequences

echo "=================================================="
echo "Causal SLM Environment Setup"
echo "=================================================="
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda not found. Please install Anaconda or Miniconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ Found conda installation"
echo ""

# Set environment name
ENV_NAME="causal-slm"

# Check if environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "⚠️  Environment '$ENV_NAME' already exists."
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n $ENV_NAME -y
    else
        echo "Exiting without changes."
        exit 0
    fi
fi

# Create conda environment
echo "Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.10 -y

# Activate environment
echo ""
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Verify activation
if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    echo "❌ Error: Failed to activate environment"
    exit 1
fi

echo "✅ Environment activated: $ENV_NAME"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (with CUDA support)
echo ""
echo "Installing PyTorch with CUDA 11.8..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install requirements
echo ""
echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

# Verify installations
echo ""
echo "=================================================="
echo "Verifying installations..."
echo "=================================================="

# Test PyTorch
python -c "import torch; print(f'✅ PyTorch {torch.__version__}'); print(f'   CUDA available: {torch.cuda.is_available()}'); print(f'   CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Test Transformers
python -c "import transformers; print(f'✅ Transformers {transformers.__version__}')"

# Test Tigramite
python -c "import tigramite; print(f'✅ Tigramite (PCMCI) installed')"

# Test DoWhy
python -c "import dowhy; print(f'✅ DoWhy {dowhy.__version__}')"

# Test NetworkX
python -c "import networkx; print(f'✅ NetworkX {networkx.__version__}')"

echo ""
echo "=================================================="
echo "GPU Configuration"
echo "=================================================="

# Detect available GPUs
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")

if [ "$GPU_COUNT" -gt 0 ]; then
    echo "✅ Found $GPU_COUNT GPU(s)"

    # Show GPU details
    python -c "
import torch
for i in range(torch.cuda.device_count()):
    print(f'   GPU {i}: {torch.cuda.get_device_name(i)}')
    print(f'      Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB')
"

    echo ""
    echo "Default GPU Configuration:"
    echo "  CUDA_VISIBLE_DEVICES=5,6,7 (GPU server default)"
    echo ""
    echo "To use specific GPUs, set environment variable:"
    echo "  export CUDA_VISIBLE_DEVICES=5,6,7"
    echo ""
    echo "Or in Python:"
    echo "  import os"
    echo "  os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'"
else
    echo "⚠️  No GPUs detected (CPU-only mode)"
    echo "   Make sure NVIDIA drivers are installed"
fi

echo ""
echo "=================================================="
echo "Setup Complete! ✅"
echo "=================================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "For GPU server, set CUDA devices:"
echo "  export CUDA_VISIBLE_DEVICES=5,6,7"
echo ""
echo "To test GPU access:"
echo "  python -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')\""
echo ""
echo "Next steps:"
echo "  1. Review MULTI_LLM_ARCHITECTURE.md"
echo "  2. Start Week 1: Literature review"
echo "  3. Download datasets (see data/README.md)"
echo ""
