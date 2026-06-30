# Training and Evaluation Guide

This document is designed to act as a standalone reference for you to train the BRAVA-GNN architecture and evaluate its performance against the exact and KADABRA-approximated algorithms, completely offline and independent of any AI assistance.

## 1. Project Dependencies

Before starting, ensure you have the following installed on your machine or remote server:
- **Julia 1.10+**: All GNN and Kadabra implementations are in Julia.
- **Python 3.10+**: Required exclusively for the synthetic training graph generation pipeline.
- **C++ Build Tools (`make`, `g++`)**: Required to compile the C++ KADABRA baseline.

## 2. Generating the Training Set

The BRAVA-GNN architecture is highly compact and size-invariant, enabling training on synthetic graphs that generalize effectively to real-world networks. The optimal training configuration established in the paper is a 30-graph mix of Directed Scale-Free, Undirected Scale-Free, and Uniformly Directed Hyperbolic Random Graphs.

To generate these exactly:

```bash
# 1. Enter the project root
cd /path/to/Lab-CA-SS26-node-importance

# 2. Set up a Python environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# 3. Install Python requirements
pip install networkit networkx

# 4. Run the data generator
python3 scripts/generate_training_data.py
```

*Note:* By default, the generator outputs $N=5000$ sized networks for quick experimentation. If you wish to run the full $N=100,000$ configuration utilized in the original paper, open `scripts/generate_training_data.py` and modify `N_NODES = 100000`.

The generated graphs will be saved as space-separated edgelists in `Instances/Training`, alongside their exact `_scores.csv` labels.

## 3. Training the BRAVA-GNN

Once the training data is generated, you can trigger the Flux.jl training loop. The script automatically fetches missing Julia dependencies via a temporary `Pkg` environment, loads the batches into a custom memory-efficient `PairwiseDataLoader`, and optimizes using the Margin Ranking Loss.

To run the training:
```bash
julia src/train_bravagnn.jl
```

To profile and time the entire execution on a headless server:
```bash
time julia src/train_bravagnn.jl
```

The script will run for 10 epochs. It calculates and outputs the loss at the end of each epoch to verify convergence. Once complete, it saves the optimized parameters locally to `bravagnn_weights.jld2`.

## 4. Benchmarking and Evaluation

We utilize a unified script to evaluate the performance of exact algorithms alongside Julia Kadabra, the original C++ Kadabra baseline, and our trained BRAVA-GNN.

To run the benchmark script:
```bash
cd benchmark
julia run_benchmarks.jl
```

### Configuration and Setup:
- **Test Files**: To add new graphs to evaluate, edit `benchmark/run_benchmarks.jl` and modify the `test_files` list array with the paths to your test datasets.
- **Model Load**: The current benchmark initiates a randomly weighted model for evaluation (to test speed). To use your serialized `bravagnn_weights.jld2` from the training phase, you should use `JLD2.@load` inside `run_benchmarks.jl` to override the initialized model before inference.
- **Ground Truth Caching**: The benchmark automatically caches the exact betweenness results so subsequent evaluation loops skip the expensive $O(|V||E|)$ ground truth calculations. 

The evaluation output is printed as a console table and dumped identically to `benchmark/benchmark_results.csv`.
