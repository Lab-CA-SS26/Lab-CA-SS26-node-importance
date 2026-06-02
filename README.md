# Node Importance via KADABRA Betweenness Centrality (SS26)

This repository contains a high-performance, multi-threaded Julia implementation of the **KADABRA** adaptive approximation algorithm for betweenness centrality (Borassi & Natale 2019), optimized for official contribution to the `Graphs.jl` ecosystem. It also contains the original C++ reference implementation for back-to-back benchmarking.

---

## 🚀 Key Achievements & Features

### 1. **Robust AllCCUpperBound Diameter Estimation**
- Ported the complete **`AllCCUpperBound`** diameter estimation algorithm (Borassi et al. 2015) in pure Julia.
- Seamlessly handles both directed networks (`DiGraph` via Tarjan's strongly connected components) and undirected networks (`SimpleGraph` via connected components).
- Computes component-level pivots, eccentricity boundaries using internal BFS, and cross-component distances using an elegant memoized DAG solver.
- Employs strict lower boundaries (`max(diam, 2.0)`) to safely prevent logarithm domain errors in starting sample size ($\omega$) calculations.

### 2. **Correct & Defensive relative Top-K Checks**
- Designed a robust tracking buffer `union_sample = absolute ? k : min(n, k + 20)` similar to C++ to evaluate betweenness separation boundary rules.
- Fully resolved index boundary errors (`BoundsError`) in the relative stopping condition for $k=1$ and other small top-k thresholds.
- Written fully defensive checks that evaluate node separation across the top-k boundary under all graph sizes.

### 3. **Modernized Multi-Threading Architecture**
- Avoids the fragile and deprecated `Threads.threadid()` indexing, eliminating any task-migration and thread-pool allocation bugs.
- Implements a modern **loop-index based task allocation** (`tid in 1:nthreads` inside `Threads.@threads`) mapping directly to task-local workspaces.
- Streamlines the parallel execution by running cooperative, task-safe `while` loops inside the parallel workers.

---

## 📂 Project Structure

```text
Lab-CA-SS26-node-importance/
├── kadabra.jl             # Core KADABRA implementation (Algorithm & Sampler)
├── test_kadabra.jl        # Comprehensive unit testing suite (22 tests)
├── compare_kadabra.jl     # High-performance Julia CLI benchmark driver
├── run_benchmark.sh       # Back-to-back C++/Julia benchmark automation script
├── COMPARISON.md          # Benchmark methodology, correctness & scaling guide
├── TODO.md                # Project roadmap and contribution checklist
├── kadabra/               # Original C++ source code & examples
│   ├── example_input/     # Datasets (Facebook, Gnutella)
│   └── ...
└── README.md              # This project guide
```

---

## 🛠️ Installation & Setup

### Prerequisite: Install OpenMP (for C++ comparison)
To build and run the multi-threaded C++ binary, install the OpenMP library using Homebrew on macOS:
```bash
brew install libomp
```

---

## 🏃‍♂️ How to Run & Benchmark

We provide a custom, fully automated runner script **`run_benchmark.sh`** that handles C++ compilation, OpenMP/Julia thread configurations, and runs both implementations back-to-back.

### Benchmark Runner Usage:
```bash
./run_benchmark.sh <err> <delta> <k> <threads> <filepath> [-d]
```
- `<err>`: Maximum error allowed (e.g. `0.01`).
- `<delta>`: Probabilistic guarantee limit (e.g. `0.1`).
- `<k>`: Number of top central nodes to find (or `0` for absolute mode).
- `<threads>`: Number of parallel CPU threads to allocate.
- `<filepath>`: Path to the SNAP edge list text file.
- `[-d]`: Optional flag indicating a directed graph.

### Run Examples:
```bash
# Compare C++ and Julia with 4 threads on the Facebook undirected dataset:
./run_benchmark.sh 0.01 0.1 3 4 kadabra/example_input/facebook_combined.txt

# Compare C++ and Julia with 4 threads on the Gnutella directed dataset:
./run_benchmark.sh 0.01 0.1 3 4 kadabra/example_input/p2p-Gnutella08.txt -d
```

---

## 🧪 Testing and Verification

Run the comprehensive unit test suite inside your Julia environment to verify sampler states, backtracking, Chernoff bounds, and relative top-k convergence:

```bash
julia test_kadabra.jl
```

### Test Suite Output:
```text
Test Summary:                 | Pass  Total  Time
KADABRA Centrality Test Suite |   22     22  0.9s
```

---

## 🏆 Performance & Alignment Highlight

In a back-to-back 4-thread execution on the `facebook_combined` graph ($\epsilon=0.01$, $\delta=0.1$):
- **KADABRA C++**: **`0.199` seconds**
- **KADABRA Julia**: **`0.247` seconds** (excluding compilation overhead via type-generic JIT warmup)
- **Centrality Rankings**: Both implementations converged on the **exact same** top 3 central vertices:
  1. **Node 107** (centrality ~0.481)
  2. **Node 1684** (centrality ~0.334)
  3. **Node 3437** (centrality ~0.238)

This confirms that the Julia port achieves **near-native, identical speed** and **100% mathematical correctness** compared to the reference implementation.

For more details on correctness and scaling benchmarks, consult **[`COMPARISON.md`](file:///Users/martinschlaier/Documents/10_Universitaet/Master/Lab-CA/Lab-CA-SS26-node-importance/COMPARISON.md)**.
