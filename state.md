# Current State of the Directory

This document summarizes the current state and structure of the `Lab-CA-SS26-node-importance` project directory.

## 1. Source Code (`src/`)
The main Julia implementations are located here:
- **`BRAVAGNN.jl`**: Contains a highly-optimized, size-invariant implementation of the BRAVA-GNN architecture. It implements a multi-hop sparse degree mass pipeline, a dual-stream message passing layer using `Flux.jl`, and a memory-efficient `PairwiseDataLoader` with a Margin Ranking Loss.
- **`kadabra.jl`**: Contains an implementation of the KADABRA algorithm for betweenness centrality approximation.

## 2. Tests (`test/` and Root)
- **`test_brava.jl`** (Root): A verification script that runs an end-to-end forward pass of `BRAVAGNN.jl` on a synthetic sparse graph to validate dimensions, flux integrations, and the pairwise loss function.
- **`test/test_kadabra.jl`**: Tests for the Kadabra implementation.
- **`test/test_kadabra_graphs_style.jl`**: Additional tests for Kadabra, seemingly oriented towards `Graphs.jl` integration.

## 3. Supporting Material & References
- **`BRAVA-GNN/`**: Contains the source research paper (`BRAVA_GNN-20.pdf`) which served as the foundation for the `BRAVAGNN.jl` model.
- **`cpp_reference/`**: Likely contains C++ reference implementations of the algorithms (e.g., Brandes or original Kadabra) used for benchmarking or correctness validation.
- **`docs/`**: Documentation files for the project.
- **`Questions.md`**: Tracks open questions, notes, or unresolved issues regarding the implementation or the lab assignment.
- **`README.md`**: Main project overview and instructions.

## 4. Evaluation & Infrastructure
- **`benchmark/`**: Directory dedicated to performance evaluation, benchmarking scripts, and comparative analysis against reference implementations.
- **`Instances/`**: Contains graph datasets and test instances used for training, benchmarking, and evaluating the node importance algorithms.
- **`venv/`**: A Python virtual environment, suggesting that some wrapper scripts, data processing, or alternative baseline implementations might be utilizing Python.

## Summary
The project is well-structured and actively tracks both classical randomized approximation algorithms (Kadabra) and modern learning-based approaches (BRAVA-GNN) in Julia. The recent additions have introduced a fully-functional GNN pipeline that is optimized for sparse operations and ready for training on the datasets present in `Instances/`.
