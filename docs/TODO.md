# KADABRA.jl Contribution to Graphs.jl - Tasks & Roadmap

This checklist tracks the implementation, bug fixes, and preparation of the KADABRA betweenness centrality algorithm in Julia for official contribution to the `Graphs.jl` ecosystem.

## 📋 Implementation & Contribution Roadmap

### Phase 1: Core Sampling Engine & Workspaces (Completed)
- [x] **State Management**: Design and implement the `KadabraWorkspace` struct to hold pre-allocated arrays (`n_paths`, `dist`, `preds`, frontier queues) to avoid dynamic allocations during sampling.
- [x] **Balanced Bidirectional BFS**: Port the frontier-aware bidirectional search with local frontier-weight heuristics to Julia (`_bb_bfs_sample!`).
- [x] **Uniform Path Selection**: Implement backtracking and randomized selection of paths across collision edge weights (`_backtrack!`).

### Phase 2: Statistical Bounds & Convergence Heuristics (Completed)
- [x] **Chernoff Bounds**: Port `compute_f` and `compute_g` mathematical bounds from Borassi & Natale (2019).
- [x] **Diameter Estimation**: Implement the `AllCCUpperBound` technique to calculate diameter bounds across connected/strongly connected components.
- [x] **Relative Top-K Stopping Condition**: Refactor `check_finished` and `kadabra_centrality` to track `union_sample` nodes, preventing `BoundsError` (such as at $k=1$) and ensuring mathematically correct top-k separation.

### Phase 3: Parallelization & Performance Optimization (Completed)
- [x] **Thread-Safe Accumulation**: Use `Threads.Atomic` for sampling and coordination counters (`n_pairs`, `stop_flag`).
- [x] **Cooperative Parallel Loop**: Streamline parallel sampling by launching a clean `while` loop inside a structured `Threads.@threads` loop across thread IDs instead of partitioning `1:typemax(Int)`.
- [x] **Memory Reuse Cleanup**: Eliminate redundant double initialization of `KadabraWorkspace` and replace `O(N log N)` sorts with `O(N log K)` `partialsortperm`.
- [x] **Zero-Allocation Pathing**: Directly increment counters during backtracking to avoid generating massive amounts of garbage.

### Phase 4: Validation & Graphs.jl Standards (Completed)
- [x] **Extended Test Suite**:
  - [x] Add path, star, and disconnected graph unit tests.
  - [x] Add explicit tests for relative top-k mode (`k > 0`) at different sizes (including $k=1$ and $k=3$) to prevent future regressions.
- [x] **Interface & API Standardization**:
  - [x] Standardize the public API to match the `Graphs.jl` centrality interface structure.
  - [x] Document all methods with clean Julia docstrings, parameter explanations, and complexity remarks.
