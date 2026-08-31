using Pkg
Pkg.activate(".")

using Graphs
using Random
using StaticGraphs

include("src/kadabra.jl")
include("benchmark/scripts/BenchmarkUtils.jl")
using .BenchmarkUtils

g_raw = BenchmarkUtils.load_graph_from_edgelist("Instances/ABCDE/com-youtube.txt", false)
g = StaticGraph(g_raw)

n = nv(g)
println("Graph (StaticGraph): nv=$n, ne=$(ne(g))")
println("Threads: $(Threads.nthreads())")
flush(stdout)

println("\nRunning kadabra_centrality on StaticGraph with parallel=true ...")
flush(stdout)

start_time = time()
res = kadabra_centrality(g, 0, 0.01, 0.1; start_factor=100, endpoints=false)
elapsed = time() - start_time

println("Done! n_samples=$(res.n_samples), time=$(round(elapsed, digits=2))s")
flush(stdout)
