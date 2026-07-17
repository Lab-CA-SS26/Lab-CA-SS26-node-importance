# Benchmark Scripts

This directory contains all the Julia scripts required to run performance evaluations, calculate exact ground truths, compare Top-K rankings, and measure parallel thread scaling.

## Available Scripts

- `run_benchmarks.jl`: The primary evaluation script. It runs KADABRA, BRAVA-GNN, and baselines against the Exact Brandes algorithm and saves metrics to `../results/benchmark_results.csv`.
- `benchmark_error_bounds.jl`: Benchmarks the error bounds of Kadabra specifically.
- `compare_topk.jl`: Tests the rank accuracy of the Top-K nodes (e.g. top 100 highest betweenness nodes) and saves to `../results/topk_results.csv`.
- `generate_benchmarks.jl`: Automates the precomputation of Exact Brandes scores. Output is stored in `../cache/`.
- `run_thread_scaling.jl`: Main entrypoint for the Thread Scaling experiments. It scales from 1 to N threads.
- `worker_thread_scaling.jl`: The worker script spawned by `run_thread_scaling.jl` to isolate Julia threading runs.
- `BenchmarkUtils.jl`: Reusable utilities and data loading functions for the benchmark scripts.
