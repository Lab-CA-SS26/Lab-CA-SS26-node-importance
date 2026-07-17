# Benchmark Comparisons

This document outlines the various benchmarks and comparisons that will be used to evaluate the implementations of Kadabra and BRAVA-GNN against Exact Brandes and each other.

## 1. Core Comparisons (Requested)

- **C++ vs Julia Kadabra**: Compare the runtime, memory usage, and output accuracy of the reference C++ implementation of Kadabra against the Julia implementation.
- **Julia Kadabra vs BRAVA-GNN (All Nodes)**: Validate the claims of the BRAVA-GNN paper by comparing the runtime and accuracy (Kendall's Tau) of BRAVA-GNN against Julia Kadabra for computing betweenness centrality on all nodes of the graph.
- **Julia Kadabra vs BRAVA-GNN (Top-k Nodes)**: Evaluate the runtime and ranking quality (Kendall's Tau, Overlap) of Julia Kadabra versus BRAVA-GNN when the goal is only to identify the top-k most central nodes for various values of `k`.
- **Kadabra Error Bounds Evaluation**: Systematically vary the error bounds (`epsilon` and `delta`) in the Julia Kadabra implementation to evaluate the trade-off between runtime and accuracy against the exact Brandes algorithm. Determine empirical values that match BRAVA-GNN's accuracy.

## 2. Additional Proposed Comparisons

- **Memory Consumption Tracking**: Profile and compare the peak memory usage of all three methods (C++ Kadabra, Julia Kadabra, and BRAVA-GNN). Memory efficiency is crucial for large-scale graph analysis.
- **Scalability Analysis (Synthetic Graphs)**: Measure how runtime and memory scale as graph size (nodes $N$ and edges $M$) and density increase. This can be evaluated by generating Barabási–Albert or Erdős–Rényi graphs of varying scales.
- **Convergence Rate vs Theoretical Bounds**: Plot the empirical number of iterations Julia Kadabra requires to converge across different `epsilon` and `delta` bounds and compare this against the theoretical maximum iteration bounds derived in the Kadabra paper.
- **Expanded Top-K Metrics (NDCG & P@k)**: Beyond simple overlap and Kendall's Tau, evaluate the top-k predictions using Information Retrieval metrics like Precision at K (P@k) and Normalized Discounted Cumulative Gain (NDCG) to better capture the quality of the ranking order within the top-k.
- **Zero-Shot Transferability**: Test BRAVA-GNN's ability to generalize to out-of-distribution graphs (e.g., training on small synthetic graphs and evaluating on large real-world social networks) and compare its robustness against Kadabra, which is training-free.
- **Heuristic Baselines**: Include fast, simple heuristics (e.g., Degree Centrality, PageRank) in the benchmarks to establish a baseline. This helps answer whether complex methods like BRAVA-GNN or Kadabra provide enough accuracy improvement over simple metrics to justify their computational overhead.
