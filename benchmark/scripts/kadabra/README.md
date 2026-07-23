# Kadabra Evaluation Pipeline

This folder contains a set of Julia scripts to evaluate and aggregate Kadabra results across various instances, implementations, threads, and seeds.

## Usage

You can run the entire pipeline at once by executing the master script from the **`benchmark`** directory:

```bash
cd benchmark
bash scripts/kadabra/run_all_evaluations.sh
```

## Available Scripts

### 1. `evaluate_kadabra_runtimes.jl`
Scans the `benchmark/output/` folder and aggregates the runtimes for each experiment variant (`instance`, `implementation`, `graph-type`, `k`, `threads`). Because you ran multiple seeds, it calculates the **mean**, **median**, and **std** of the `execution_time_seconds` across all seeds.
* **Output:** `benchmark/results/kadabra/kadabra_runtimes_summary.csv`

### 2. `evaluate_kadabra_speedup.jl`
Reads the runtimes summary generated above and automatically compares `kadabra-julia` against `kadabra-cpp` for matching configs. It computes the speedup as $T_{cpp} / T_{julia}$ (values $> 1$ mean Julia is faster).
* **Output:** `benchmark/results/kadabra/kadabra_speedup_comparison.csv`

### 3. `evaluate_kadabra_optimal.jl`
Reads the runtimes summary and searches for the optimal (minimum) runtime configuration (the best number of threads) for both C++ and Julia on each graph instance. It then calculates the ratio of the best C++ time to the best Julia time.
* **Output:** `benchmark/results/kadabra/kadabra_optimal_configs_comparison.csv`

### 4. `evaluate_kadabra_accuracy.jl`
Searches `../Instances/` for exact ground-truth scores (files ending in `-score.txt`). Then, for each matching Kadabra experiment in `output/`, it calculates:
- **Global Kendall Tau** (Rank correlation across all nodes)
- **Top-10 and Top-100 Kendall Tau** (Rank correlation strictly for the true top nodes)
- **Top-10 and Top-100 Overlap** (How many of the exact top nodes are found in Kadabra's top nodes)
- **Max Rank Required** (How far down Kadabra's list you must look to find *all* true top nodes)
- **Candidate Set Size** (For $k > 0$, the exact number of nodes Kadabra's bounding process returned because their upper bounds were $\ge$ the $k$-th highest lower bound)

It aggregates these metrics by calculating the **mean** and **standard deviation** across all seeds for a robust accuracy profile.
* **Output:** `benchmark/results/kadabra/kadabra_accuracy_summary.csv`
