# Benchmark Results

This directory contains the aggregated output metrics from the various benchmarking scripts.

## Core Outputs
- `benchmark_results.csv`: Contains the master list of speed, memory, and Kendall Tau comparisons for Exact, KADABRA, BRAVA-GNN, and baseline algorithms.
- `topk_results.csv`: Contains the evaluation of algorithms solely based on predicting the highest-betweenness nodes.
- `exact_brandes_timings.csv`: Records baseline timings of the $O(|V||E|)$ Exact algorithm.
- `thread_scaling_results.csv`: Records metrics for multi-threaded Kadabra execution scaling runs.
