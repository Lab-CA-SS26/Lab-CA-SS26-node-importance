# Benchmarks

This folder contains the experimental setup to benchmark the runtime and accuracy of the various betweenness centrality algorithms (Julia Kadabra, C++ Kadabra, BRAVA-GNN, and Exact Brandes).

We use **[simexpal](https://github.com/hu-macsy/simexpal)** to cleanly manage large-scale execution on compute clusters, alongside JSON logging for precise and unified metric tracking.

## Supported Algorithms
The `simexpal` pipeline (`experiments.yml`) currently evaluates:
- `kadabra-cpp`: Original Kadabra implementation in C++
- `kadabra-julia`: Our custom Julia Kadabra implementation
- `brava-julia`: GNN-based inference (requires `.jld2` weights inside `cache/`)
- `brandes-julia`: Exact algorithm serving as the ground-truth baseline (via `Graphs.jl`)

## Execution Metrics
Each execution yields a standardized `stats.json` file inside the `output/` directory containing:
- `centralities`: Dictionary mapping `node_id -> betweenness_score`
- `execution_time_seconds`: Pure algorithmic computation time (fair comparison)
- `io_time_seconds`: Graph parsing and memory loading time
- `num_samples`: Algorithm sample counts (e.g., pairs used for Kadabra)
- `parameters`: Metadata (threads, graph parameters, epsilon, etc.)

---

## Managing Experiments with `simexpal`

All benchmarks are orchestrated through `experiments.yml` and launched via the Python wrapper `simex`. It is recommended to use the binary located in your `venv/`.

### 1. Launching
Build necessary programs and launch all pending experiments:
```bash
cd simexpal_runners && make build && cd ..
../venv/bin/simex e launch
```

### 2. Checking Status
View the status of experiments (running, finished, failed):
```bash
../venv/bin/simex e list
```

### 3. Purging (Deleting Runs)
If you need to delete and re-run experiments, use the `purge` command with `-f` (force).

**Purge everything:**
```bash
../venv/bin/simex e purge --all -f
```

**Purge a specific algorithm/experiment:**
```bash
../venv/bin/simex e purge --experiment brandes-julia -f
```

**Purge a specific graph instance:**
```bash
../venv/bin/simex e purge --instance p2p-Gnutella31 -f
```

**Purge a combination of Experiment and Instance:**
```bash
../venv/bin/simex e purge --experiment brandes-julia --instance p2p-Gnutella31 -f
```

**Purge all failed runs:**
*(Useful for restarting only crashed executions)*
```bash
../venv/bin/simex e purge --failed -f
```

**Purge an exact run string:**
```bash
../venv/bin/simex e purge --run "brandes-julia~err1,k0,t1,undirected/p2p-Gnutella31[0]" -f
```

---

## Legacy Scripts
Earlier iterations of local benchmarking can still be found in the root directory:
- `run_benchmarks.jl`
- `compare_topk.jl`
- `benchmark_error_bounds.jl`
- `generate_benchmarks.jl`
