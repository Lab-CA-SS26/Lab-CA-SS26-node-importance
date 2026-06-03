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

---

## 📂 File Directory

- **`kadabra.jl`**: Core Julia implementation containing the sampler, BFS workspace, and mathematical bounds.
- **`test_kadabra.jl`**: Automated unit test suite verifying correctness, sampling, and convergence.
- **`compare_kadabra.jl`**: Julia command-line driver script for loading SNAP datasets and running benchmarks.
- **`run_benchmark.sh`**: Bash script to compile the C++ binary and run both implementations back-to-back under identical thread counts.
- **`COMPARISON.md`**: Guide describing the benchmark parameters, execution workflow, and results.
- **`ROADMAP.md`**: Guide detailing instructions for contributing code to the open-source `Graphs.jl` library.

---

## 🏃‍♂️ How to Run Benchmarks

The benchmark runner script compiles the C++ program (if needed), configures threads, and runs both versions.

### Usage:
```bash
./run_benchmark.sh <err> <delta> <k> <threads> <filepath> [-d]
```
*Options:*
- `-d`: Flag indicating a directed graph (must match the input dataset).

### Examples:
```bash
# Benchmark 4 threads on the Facebook undirected dataset:
./run_benchmark.sh 0.01 0.1 3 4 kadabra/example_input/facebook_combined.txt

# Benchmark 4 threads on the Gnutella directed dataset:
./run_benchmark.sh 0.01 0.1 3 4 kadabra/example_input/p2p-Gnutella08.txt -d
```

---

## 🧪 How to Run Tests

Run the automated test suite to verify the sampler, backtracking, and convergence criteria:

```bash
julia test_kadabra.jl
```
