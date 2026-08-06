#!/bin/bash
set -e

echo "======================================"
echo "    KADABRA/BRAVA Benchmarking Pipeline"
echo "======================================"

RUN_SIMEXPAL=true

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --no-run|-n) RUN_SIMEXPAL=false; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

# Ensure we are in the root directory
cd "$(dirname "$0")"

# 1. Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "[1/4] Activating Python virtual environment..."
    source venv/bin/activate
else
    echo "[1/4] No venv found. Using system python."
fi

# 2. Run Experiments via Simexpal
if [ "$RUN_SIMEXPAL" = true ]; then
    echo "[2/4] Launching simexpal experiments..."
    cd benchmark
    
    if ! command -v simexpal &> /dev/null; then
        echo "Error: simexpal could not be found. Please install it or use --no-run."
        exit 1
    fi
    
    simexpal launch
    cd ..
else
    echo "[2/4] Skipping simexpal experiments (--no-run flag passed)."
fi

# 3. Evaluate runs
echo "[3/4] Evaluating all runs..."
julia --project=. evaluate_all_runs.jl

# 4. Generate plots
echo "[4/4] Generating plots..."
if [ -f "plot_results.py" ]; then
    python3 plot_results.py
else
    echo "plot_results.py not found. Skipping default plot generation."
fi

if [ -f "plot_kadabra_julia.py" ]; then
    echo "Generating Julia-specific plots..."
    python3 plot_kadabra_julia.py
else
    echo "plot_kadabra_julia.py not found. Skipping."
fi

echo "======================================"
echo "    Pipeline completed successfully!  "
echo "    Results and plots are in results/ "
echo "======================================"
