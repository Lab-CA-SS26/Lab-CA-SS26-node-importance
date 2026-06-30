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
- **`src/train_bravagnn.jl`**: The training script that applies the Margin Ranking Loss.
- **`scripts/generate_training_data.py`**: A Python script to generate the synthetic graphs (Directed/Undirected Scale-Free and Hyperbolic Random Graphs) and exact labels.
- **`test/`**: Contains `test_kadabra_graphs_style.jl`, the automated unit test suite verifying correctness, sampling, and convergence against the Graphs.jl ecosystem.
- **`benchmark/`**: Contains all benchmarking utilities:
  - `run_benchmarks.jl`: Unified evaluation script comparing Exact Betweenness, Julia KADABRA, C++ KADABRA, and BRAVA-GNN.
  - `BenchmarkUtils.jl`: Module handling C++ interoperability and parsing.
- **`cpp_reference/`**: The original Borassi C++ implementation for baseline comparisons.
- **`docs/`**: Additional reading materials:
  - `COMPARISON.md`: Guide describing the benchmark parameters and results.
  - `ROADMAP.md`: Guide detailing instructions for contributing code to `Graphs.jl`.

---

## 🏃‍♂️ How to Run Benchmarks

The benchmark runner script (`benchmark/run_benchmarks.jl`) compiles the C++ program (if needed), loads the configured test graphs, and runs exact calculation, KADABRA (C++ and Julia), and BRAVA-GNN, and outputs the statistics to a CSV file.

### Usage:
```bash
cd benchmark
julia run_benchmarks.jl
```
You can easily expand the evaluated datasets by appending them into the `test_files` list array inside the script.

---

## 📝 TODO / Next Steps

- [ ] **Train BRAVA-GNN**: The current `run_benchmarks.jl` pipeline executes a model with random weights. We need to implement a full training script that generates training pairs using `PairwiseDataLoader` and applies the Margin Ranking Loss.
- [ ] **Scale Benchmarking**: Add a wider array of `Instances/` networks and optionally test memory limitations across scale up to $1M+$ nodes.
- [ ] **Heuristic Pruning Integration**: Consider integrating heuristic graph pruning to speed up exact Brandes calculation used for ground truth labels.
- [ ] **Continuous Integration**: Setup GitHub actions to automate tests and benchmarks against C++ reference on pushes.
