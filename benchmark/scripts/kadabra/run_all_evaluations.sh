#!/bin/bash

# Make sure we are in the benchmark directory
if [ ! -d "scripts/kadabra" ]; then
    echo "Please run this script from the benchmark directory:"
    echo "  cd benchmark"
    echo "  bash scripts/kadabra/run_all_evaluations.sh"
    exit 1
fi

echo "========================================="
echo "  Kadabra Evaluation Pipeline"
echo "========================================="

echo ""
echo "[1/4] Aggregating Runtimes..."
julia scripts/kadabra/evaluate_kadabra_runtimes.jl

echo ""
echo "[2/4] Calculating Speedup..."
julia scripts/kadabra/evaluate_kadabra_speedup.jl

echo ""
echo "[3/4] Calculating Optimal Configs..."
julia scripts/kadabra/evaluate_kadabra_optimal.jl

echo ""
echo "[4/4] Evaluating Accuracy (Kendall Tau & Overlap)..."
julia scripts/kadabra/evaluate_kadabra_accuracy.jl

echo ""
echo "========================================="
echo "  Pipeline Complete!"
echo "  All results saved in benchmark/results/kadabra/"
echo "========================================="
