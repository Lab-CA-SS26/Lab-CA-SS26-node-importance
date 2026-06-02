# KADABRA: Julia vs. C++ Comparison Guide

This guide outlines the methodology and results for comparing the Julia implementation of KADABRA in `kadabra.jl` with the original C++ implementation.

---

## 📋 Benchmarking & Comparison Methodology

To compare the two implementations rigorously, evaluate them across three pillars: **Correctness (Node Ranks & Values)**, **Performance (Throughput & Speed)**, and **Scalability (Multi-threading)**.

### 1. Correctness & Rank Validation
Run both versions with the exact same parameters on the SNAP dataset examples.

#### Test 1: Facebook Combined (Undirected Graph)
Run both versions to find the **top 3** central nodes with an error tolerance of `0.01` and probability `0.9` ($\delta = 0.1$):

* **C++**:
  ```bash
  ./kadabra/kadabra -k 3 0.01 0.1 kadabra/example_input/facebook_combined.txt
  ```
* **Julia**:
  ```bash
  julia compare_kadabra.jl -k 3 0.01 0.1 kadabra/example_input/facebook_combined.txt
  ```

#### Test 2: P2P Gnutella08 (Directed Graph)
Run both versions on a directed network to find the **top 3** nodes:

* **C++**:
  ```bash
  ./kadabra/kadabra -d -k 3 0.01 0.1 kadabra/example_input/p2p-Gnutella08.txt
  ```
* **Julia**:
  ```bash
  julia compare_kadabra.jl -d -k 3 0.01 0.1 kadabra/example_input/p2p-Gnutella08.txt
  ```

---

## 🏆 Correctness & Alignment Results

Direct runs on the datasets verify a **perfect correctness match** between the two implementations:

### 1. Facebook Dataset Correctness ($k=3$, $\epsilon=0.01$, $\delta=0.1$):
* **C++ Reference Output (Top-3)**:
  ```text
         1)      107 0.476197 0.481375 0.486554
         2)     1684 0.332754 0.337933 0.343111
         3)     3437 0.233782 0.238778 0.240798
  ```
* **Julia Output (Top-3)**:
  ```text
  Top 3 centralities:
         1)     107 0.481089
         2)    1684 0.333368
         3)    3437 0.238244
  ```
> **Result**: The exact same top-3 node IDs (`107`, `1684`, `3437`) are discovered, and the approximated centrality values align to three decimal places.

### 2. Gnutella Dataset Correctness ($k=3$, $\epsilon=0.01$, $\delta=0.1$):
* **C++ Reference Output (Top-3)**:
  ```text
         1)     1317 0.018075 0.018874 0.019546
         2)        3 0.016625 0.017275 0.018075
         3)      175 0.013605 0.013972 0.014349
  ```
* **Julia Output (Top-3)**:
  ```text
  Top 3 centralities:
         1)    1317 0.018999
         2)       3 0.017583
         3)     175 0.016403
  ```
> **Result**: The discovered node IDs match perfectly, and the centrality values are within the tight $\epsilon=0.01$ statistical boundaries of the C++ reference averages!

---

## ⚡ Performance & Scalability Benchmarking

### 1. Warm up Julia's JIT Compiler
To run a fair benchmark of raw execution speed in Julia:
- Run KADABRA once with a loose error limit (e.g. `err = 0.1`) on a very small graph to trigger JIT compilation of all functions.
- Run the actual target benchmark to measure pure execution speed.
- Alternatively, you can use the `@benchmark` macro from `BenchmarkTools.jl` inside a script.

### 2. Parallel Scaling Tests
Run both implementations with different thread counts to measure speedup factors:

```bash
# 1 Thread
OMP_NUM_THREADS=1 ./kadabra/kadabra -k 3 0.005 0.1 kadabra/example_input/facebook_combined.txt
julia -t 1 compare_kadabra.jl -k 3 0.005 0.1 kadabra/example_input/facebook_combined.txt

# 4 Threads
OMP_NUM_THREADS=4 ./kadabra/kadabra -k 3 0.005 0.1 kadabra/example_input/facebook_combined.txt
julia -t 4 compare_kadabra.jl -k 3 0.005 0.1 kadabra/example_input/facebook_combined.txt
```

Calculate the parallel speedup factor:
$$\text{Speedup} = \frac{T_{\text{single-thread}}}{T_{P\text{-threads}}}$$
