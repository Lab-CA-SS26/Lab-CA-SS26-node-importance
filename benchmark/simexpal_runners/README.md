# Simexpal Runners

This directory accumulates the execution entry points for running KADABRA and BRAVA experiments using `simexpal`. 
By centralizing the runners, the experiment configuration stays clean and maintainable.

## Files
- `run_experiments.cpp`: The native C++ wrapper for the `Kadabra` implementation. Parses arguments, tracks OpenMP threads, calls `Probabilistic::run` and serializes the exact matching output into JSON.
- `run_experiments.jl`: The dynamic Julia driver for testing both the Julia `kadabra` translation and `BRAVAGNN` inference.
- `Makefile`: A customized build script that links `run_experiments.cpp` against the `cpp_reference` source codebase located in the root directory.

## Expected JSON Schema
Both wrappers adhere to the following output format for `simexpal` metrics:
```json
{
  "parameters": {
    "input_file": "<path>",
    "algorithm": "kadabra",
    "version": "julia",
    "threads": 4,
    "k": 10,
    "delta": 0.1,
    "epsilon": 0.01,
    "directed": false
  },
  "execution_time_seconds": 1.234,
  "num_samples": 120500,
  "centralities": {
    "1": 0.0452,
    "2": 0.0123
  }
}
```

## Running
Simexpal automatically handles compiling and launching these scripts.
```bash
# In the benchmark directory:
simex b remake
simex e launch
```
