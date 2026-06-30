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
- **`src/train_bravagnn.jl`**: The full training pipeline loop utilizing Pairwise Margin Ranking loss and offline exact-BC data loading.
- **`scripts/generate_training_data.py`**: An authentic adaptation of the BRAVA-GNN original generation script. Builds Scale-Free and Hyperbolic topological instances with raw text edge and score exports.
- **`scripts/download_datasets.py`**: An automated downloader for the 14 real-world test and calibration graphs from the SNAP/ABCDE repositories.
- **`test/`**: Contains automated unit tests verifying KADABRA correctness against Graphs.jl.
- **`benchmark/`**: Contains benchmarking suites:
  - `run_benchmarks.jl`: Base execution script assessing inference time and absolute error across algorithms.
  - `compare_topk.jl`: Specialized top-$k$ evaluation script calculating Kendall Tau and Set Overlap between KADABRA and BRAVA-GNN.
  - `BenchmarkUtils.jl`: Module handling C++ interoperability and parsing.
- **`cpp_reference/`**: The original Borassi C++ implementation for baseline comparisons.
- **`docs/`**: Additional reading materials:
  - `COMPARISON.md`: Guide describing the benchmark parameters and results.
  - `ROADMAP.md`: Guide detailing instructions for contributing code to `Graphs.jl`.

---

## 📦 How to get the Data

All benchmark evaluation graphs and synthetic training topologies are completely reproducible. First, ensure your Python virtual environment is activated and `networkit` is installed.

To fetch the 14 real-world calibration and test graphs natively into `Instances/TestInstances/`:
```bash
source venv/bin/activate
python scripts/download_datasets.py --calibration
```

To natively generate the synthetic structural data required to train BRAVA-GNN, exported as text edge-lists to `Instances/Training/`:
```bash
python scripts/generate_training_data.py --datasets SF_10_Dir SF_10_Sym HY_10_Dir --num_nodes 100000
```

---

## 🏃‍♂️ Training & Experiments

### Training BRAVA-GNN
Once your training graphs exist in `Instances/Training/`, you can train the PyTorch-equivalent Flux.jl weights. Make sure to launch this on a server utilizing multiple Julia threads.
```bash
julia --threads=auto src/train_bravagnn.jl
```
This produces a `bravagnn_weights.jld2` artifact inside `benchmark/`.

### Top-k Ranking Evaluation
To evaluate whether the adaptive probabilistic halting of KADABRA outperforms the fixed inference pass of the trained BRAVA-GNN on the test set, run the top-k comparison. It will measure ranking accuracy natively using **Kendall Tau** and **Set Overlap**:
```bash
cd benchmark
julia --project compare_topk.jl
```

---

## 📝 Next Steps

- [ ] **Scale Benchmarking**: Add a wider array of `Instances/` networks and optionally test memory limitations across scale up to $1M+$ nodes.
- [ ] **Heuristic Pruning Integration**: Consider integrating heuristic graph pruning to speed up exact Brandes calculation used for ground truth labels.
- [ ] **Continuous Integration**: Setup GitHub actions to automate tests and benchmarks against C++ reference on pushes.
