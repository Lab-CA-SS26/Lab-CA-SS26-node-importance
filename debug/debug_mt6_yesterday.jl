using Pkg
Pkg.activate(".")
using Graphs, StaticGraphs, Random

include("kadabra_yesterday.jl")
include("benchmark/scripts/BenchmarkUtils.jl")
using .BenchmarkUtils

println("Loading graph...")
flush(stdout)
t0 = time()
g_raw = BenchmarkUtils.load_graph_from_edgelist("Instances/ABCDE/com-youtube.txt", false)
g = StaticGraph(g_raw)
println("Graph loaded in $(round(time()-t0, digits=2))s: nv=$(nv(g)), ne=$(ne(g))")
flush(stdout)

# JIT warmup like the benchmark runner does
println("\nJIT Warmup (same as run_experiments.jl)...")
flush(stdout)
t1 = time()
dummy_g_raw = SimpleGraph{Int64}(3)
add_edge!(dummy_g_raw, 1, 2)
add_edge!(dummy_g_raw, 2, 3)
dummy_g = StaticGraph(dummy_g_raw)
kadabra_centrality(dummy_g, 0, 0.01, 0.1; start_factor=10, endpoints=false)
println("JIT Warmup done in $(round(time()-t1, digits=2))s")
flush(stdout)

# Actual run
println("\nActual kadabra run on com-youtube with $(Threads.nthreads()) threads...")
flush(stdout)
t2 = time()
res = kadabra_centrality(g, 0, 0.01, 0.1; start_factor=100, endpoints=false)
println("Done in $(round(time()-t2, digits=2))s, n_samples=$(res.n_samples)")
flush(stdout)
