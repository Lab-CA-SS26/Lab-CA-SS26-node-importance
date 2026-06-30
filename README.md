# KADABRA Betweenness Centrality (SS26)

This repository contains a high-performance, multi-threaded Julia implementation of the **KADABRA** adaptive approximation algorithm for betweenness centrality (Borassi & Natale 2019), optimized for the `Graphs.jl` ecosystem. It also includes the C++ reference implementation and a back-to-back benchmarking suite.

---

## 🚀 Features & Implementation Details

### 1. AllCCUpperBound Diameter Estimation
- Computes a tight upper bound on the graph's diameter using the **AllCCUpperBound** technique (Borassi et al. 2015).
- Supports both directed graphs (`DiGraph` via strongly connected components) and undirected graphs (`SimpleGraph` via connected components).
- Leverages component pivots and memoized DAG traversal to bound the initial sample size ($\omega$) efficiently.

### 2. Top-K Convergence Checking
- Features an adaptive stopping rule that evaluates Chernoff boundaries dynamically during sampling.
- Supports both absolute error approximation ($k=0$) and relative top-k ranking ($k > 0$).
- Tracks a sample buffer of size `min(n, k + 20)` to ensure mathematically correct rank separation at the top-k boundary.

### 3. Multi-Threaded Sampling Engine
- Utilizes task-local workspaces mapped to thread workers via contiguous task indices.
- Safe against task migration and dynamic thread-pool scaling in Julia.
- Implements concurrent sampling loops coordinated via thread-safe atomic counters (`n_pairs` and `stop_flag`).

### 4. Full Graphs.jl API Compatibility
- Implements the exact same `normalize` and `endpoints` keyword arguments as `Graphs.jl`'s standard `betweenness_centrality`.
- By default, KADABRA's outputs are perfectly scaled using the standard `Graphs.jl` normalization scalars for directed/undirected graphs, rather than the raw probabilistic expected values output by the original C++ reference paper.

---

## 📂 File Directory

The repository is structured as follows:

- **`src/kadabra.jl`**: Core Julia implementation of KADABRA. Fully optimized with zero-allocation path sampling and O(N log K) convergence checking.
- **`src/BRAVAGNN.jl`**: Implementation of the BRAVA-GNN architecture for scalable betweenness centrality estimation using SparseArrays and Flux.jl.
- **`test/`**: Contains `test_kadabra_graphs_style.jl`, the automated unit test suite verifying correctness, sampling, and convergence against the Graphs.jl ecosystem.
- **`benchmark/`**: Contains all benchmarking utilities:
  - `run_benchmarks.jl`: Unified evaluation script comparing Exact Betweenness, Julia KADABRA, C++ KADABRA, and BRAVA-GNN.
  - `BenchmarkUtils.jl`: Module handling C++ interoperability and parsing.
  - `compare_kadabra.jl`: Julia command-line driver script for loading SNAP datasets.
  - `run_benchmark.sh`: Bash script to compile the C++ binary and run both implementations back-to-back.
  - `experiments.yml`: Simexpal configuration matrix for structured benchmarks.
- **`cpp_reference/`**: The original Borassi C++ implementation for baseline comparisons.
- **`docs/`**: Additional reading materials:
  - `COMPARISON.md`: Guide describing the benchmark parameters and results.
  - `ROADMAP.md`: Guide detailing instructions for contributing code to `Graphs.jl`.

---

## 🏃‍♂️ How to Run Benchmarks

The benchmark runner script compiles the C++ program (if needed), configures threads, and runs both versions.

### Usage:
```bash
cd benchmark
./run_benchmark.sh <err> <delta> <k> <threads> <filepath> [-d]
```
*Options:*
- `-d`: Flag indicating a directed graph (must match the input dataset).

### Examples:
```bash
# Benchmark 4 threads on the Facebook undirected dataset:
./run_benchmark.sh 0.01 0.1 3 4 ../Instances/SNAP_Instances/facebook_combined.txt

# Benchmark 4 threads on the Gnutella directed dataset:
./run_benchmark.sh 0.01 0.1 3 4 ../Instances/SNAP_Instances/p2p-Gnutella08.txt -d
```

---

## 🧪 How to Run Tests

Run the automated test suite to verify the sampler, backtracking, and convergence criteria. Make sure to run it with the `Graphs.jl` test environment active:

```bash
julia --project=../Graphs.jl/test test/test_kadabra_graphs_style.jl
```

---

## 📊 Algorithmic Experiments with simexpal

We have integrated a declarative experimental setup using **`simexpal`** (an algorithmic experiment management tool developed by the MACSy group at HU Berlin). This allows you to automatically download datasets, compile code, and run structured benchmark combinations (cross-product of threads and $k$-values) with a single command.

The configuration is defined in **[`experiments.yml`](file:///Users/martinschlaier/Documents/10_Universitaet/Master/Lab-CA/Lab-CA-SS26-node-importance/experiments.yml)**.

### 1. Installation
Install the `simexpal` package using pip:
```bash
pip install simexpal
```

### 2. Verify Local Instances
Use `simexpal` to verify that your local SNAP graph datasets are successfully detected inside your `Instances/SNAP_Instances` directory:
```bash
# List all configured instances (should show green/available)
simex instances list
```

### 3. Run the Benchmark Matrix
You can run and track all experiment permutations (C++ and Julia, 1 vs 4 threads, $k=3$ vs $k=10$):
```bash
# Display the combinations in the run matrix
simex matrix

# Execute all experiments
simex run
```

---

## 📝 TODO / Next Steps

- [ ] **Train BRAVA-GNN**: The current `run_benchmarks.jl` pipeline executes a model with random weights. We need to implement a full training script that generates training pairs using `PairwiseDataLoader` and applies the Margin Ranking Loss.
- [ ] **Scale Benchmarking**: Add a wider array of `Instances/` networks and optionally test memory limitations across scale up to $1M+$ nodes.
- [ ] **Heuristic Pruning Integration**: Consider integrating heuristic graph pruning to speed up exact Brandes calculation used for ground truth labels.
- [ ] **Continuous Integration**: Setup GitHub actions to automate tests and benchmarks against C++ reference on pushes.
