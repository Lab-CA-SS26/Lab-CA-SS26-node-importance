# Benchmarks

This folder contains scripts and utilities designed to benchmark the runtime, memory consumption, and accuracy of the various betweenness centrality algorithms (Julia Kadabra, C++ Kadabra, BRAVA-GNN, and Exact Brandes).

Key scripts include:
- `run_benchmarks.jl`: General benchmarking of execution time and accuracy on all nodes.
- `compare_topk.jl`: Evaluating precision and NDCG for Top-k node identification.
- `benchmark_error_bounds.jl`: Analyzing the tradeoff between approximation bounds (`epsilon`, `delta`) and runtime.
- `generate_benchmarks.jl`: Generates and caches exact Brandes scores for graphs in the `Instances` folder.
