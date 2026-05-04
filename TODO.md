# Julia Implementation of KADABRA for Graphs.jl

This document outlines the strategy for porting the KADABRA betweenness centrality algorithm from C++ to Julia, intended for contribution to the `Graphs.jl` ecosystem.

## 📋 Implementation Roadmap

### Phase 1: Environment & Standards Research
- [ ] Study existing centrality implementations in `Graphs.jl`.
- [ ] Define the `AbstractGraph` and `AbstractDiGraph` interface support.
- [ ] Determine the best approach for multi-threading (Threads vs. Atomics).

### Phase 2: Core Algorithm Porting (The "Engine")
- [ ] **Bidirectional BFS:** Implement the frontier-aware search in Julia.
- [ ] **State Management:** Design a `KadabraState` struct to hold `n_paths`, `dist`, and visit indicators.
- [ ] **Path Sampling:** Implement the uniform random path selection and backtracking logic.

### Phase 3: Statistical Bounds (The "Brain")
- [ ] **Chernoff Bounds:** Port `compute_f` and `compute_g` logic.
- [ ] **Diameter Estimation:** Implement the `AllCCUpperBound` technique for diameter upper-bounding.
- [ ] **Convergence Logic:** Implement the stopping condition based on epsilon and delta.

### Phase 4: Parallelization & Optimization
- [ ] **Thread-Safe Counters:** Use `Threads.Atomic` for centrality accumulation.
- [ ] **Memory Reuse:** Pre-allocate thread-local buffers to minimize garbage collection overhead.
- [ ] **Performance Profiling:** Use `BenchmarkTools.jl` and `@code_warntype` for optimization.

### Phase 5: Validation & Integration
- [ ] **Unit Tests:** Verify the sampler on small known graphs (cycle, star, path).
- [ ] **Cross-Validation:** Compare results with the C++ implementation on `facebook_combined.txt`.
- [ ] **Documentation:** Write docstrings and examples in line with `Graphs.jl` standards.

---

## 🏗️ Proposed Folder Structure

```
GraphsKADABRA.jl/
├── src/
│   ├── GraphsKADABRA.jl  # Main module entry and public API
│   ├── sampler.jl        # Bidirectional BFS and path sampling
│   ├── bounds.jl         # Statistical calculations and stopping rules
│   └── utils.jl          # Diameter estimation and helper functions
├── test/
│   └── runtests.jl       # Comprehensive test suite
├── docs/                 # Documentation and usage examples
├── Project.toml          # Julia package dependencies
└── README.md             # Project overview and installation
```
