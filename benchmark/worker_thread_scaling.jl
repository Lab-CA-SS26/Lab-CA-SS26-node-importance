# Worker script for thread scaling
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using Graphs
using Printf

include(joinpath(@__DIR__, "..", "src", "kadabra.jl"))
include(joinpath(@__DIR__, "BenchmarkUtils.jl"))

if length(ARGS) < 2
    error("Please provide the dataset path relative to Instances/ and a flag D/U")
end

dataset_name = ARGS[1]
is_directed = ARGS[2] == "D"
full_path = joinpath(@__DIR__, "..", "Instances", dataset_name)
nthreads = Threads.nthreads()

# 1. Load graph
g = BenchmarkUtils.load_graph_from_edgelist(full_path, is_directed)

# Parameters for Kadabra
k = 0 # absolute error for all nodes
epsilon = 0.01
delta = 0.1

# 2. Compilation warm-up (so TTFX doesn't ruin the timings)
dummy_g = path_graph(10)
kadabra_centrality(dummy_g, k, epsilon, delta)

# 3. Run benchmark
start_time = time()
res = kadabra_centrality(g, k, epsilon, delta)
elapsed = time() - start_time
@printf("     Worker finished. Threads: %2d | Time: %.2f seconds\n", nthreads, elapsed)

# 4. Write result to the shared CSV file
out_csv = joinpath(@__DIR__, "thread_scaling_results.csv")
open(out_csv, "a") do io
    println(io, "$dataset_name,$nthreads,$elapsed")
end
