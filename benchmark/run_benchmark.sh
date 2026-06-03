#!/bin/bash

# Exit on error
set -e

# Usage helper
if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <err> <delta> <k> <threads> <filepath> [-d]"
    echo "Options:"
    echo "  -d    Graph is directed (default: undirected)"
    exit 1
fi

ERR=$1
DELTA=$2
K=$3
THREADS=$4
FILEPATH=$5

# Directed graph flags
DIRECTED_CPP=""
DIRECTED_JL=""
if [ "$6" == "-d" ]; then
    DIRECTED_CPP="-d"
    DIRECTED_JL="-d"
fi

# Locate directories and scripts
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
CPP_BIN="$BASE_DIR/../cpp_reference/kadabra"
CPP_DIR="$BASE_DIR/../cpp_reference"
JL_SCRIPT="$BASE_DIR/compare_kadabra.jl"

# 1. Compile C++ if binary is missing
if [ ! -f "$CPP_BIN" ]; then
    echo "==> C++ binary not found. Compiling now..."
    cd "$CPP_DIR"
    g++ -Xpreprocessor -fopenmp -std=c++11 -O3 -Wall -g -Iinclude -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp main.cpp src/* -lm -o kadabra
    cd "$BASE_DIR"
    echo "==> C++ binary compiled successfully."
fi

# Format output
echo "========================================================================"
echo "                  KADABRA BACK-TO-BACK COMPARISON                       "
echo "========================================================================"
echo "Parameters:"
echo "  • Dataset:     $(basename "$FILEPATH")"
echo "  • Error (err): $ERR"
echo "  • Delta (δ):   $DELTA"
echo "  • Top-K (k):   $K"
echo "  • Threads:     $THREADS"
echo "  • Directed:    $( [ -n "$DIRECTED_CPP" ] && echo "Yes" || echo "No" )"
echo "========================================================================"
echo ""

# 2. Execute C++ KADABRA
echo "🔥 Running KADABRA C++ (OpenMP)..."
echo "------------------------------------------------------------------------"
OMP_NUM_THREADS=$THREADS "$CPP_BIN" $DIRECTED_CPP -k $K $ERR $DELTA "$FILEPATH"
echo "------------------------------------------------------------------------"
echo ""

# 3. Execute Julia KADABRA
echo "🔥 Running KADABRA Julia (Threads)..."
echo "------------------------------------------------------------------------"
julia -t $THREADS "$JL_SCRIPT" $DIRECTED_JL -k $K $ERR $DELTA "$FILEPATH"
echo "------------------------------------------------------------------------"
echo "========================================================================"
