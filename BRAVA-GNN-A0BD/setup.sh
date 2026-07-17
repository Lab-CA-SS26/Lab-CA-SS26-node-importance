#!/bin/bash
# Set up a Python virtual environment for BRAVA-GNN.
# Usage:  source setup.sh
#
# Installs PyTorch + PyG (CUDA 11.8 wheels) and the requirements in requirements.txt,
# then installs the ABCDE baseline as an editable package.

safe_exit() {
    if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then return "$1"; else exit "$1"; fi
}

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv &> /dev/null; then
    python3 -m pip install uv --user || safe_exit 1
fi

if [ ! -d ".venv" ]; then
    if command -v python3.11 &> /dev/null; then
        uv venv --python python3.11 .venv || safe_exit 1
    else
        uv venv --python python3 .venv || safe_exit 1
    fi
fi

source .venv/bin/activate

uv pip install \
    "torch==2.3.1+cu118" \
    "torchvision==0.18.1+cu118" \
    "torchaudio==2.3.1+cu118" \
    --index-url https://download.pytorch.org/whl/cu118 \
    --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    ninja packaging setuptools wheel || safe_exit 1

uv pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    --find-links https://data.pyg.org/whl/torch-2.3.1+cu118.html \
    --no-build-isolation || safe_exit 1

if [ -f "requirements.txt" ]; then
    uv pip install -r requirements.txt || safe_exit 1
fi

if [ -d "baselines/abcde" ]; then
    uv pip install -e baselines/abcde --no-deps || safe_exit 1
fi

echo "Done. Environment ready."
