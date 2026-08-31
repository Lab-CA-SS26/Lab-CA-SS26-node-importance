using Pkg
Pkg.activate(".")
include("benchmark/scripts/BenchmarkUtils.jl")
using .BenchmarkUtils
include("src/kadabra.jl")
g_raw = BenchmarkUtils.load_graph_from_edgelist("Instances/ABCDE/com-youtube.txt", false)
g = g_raw

println("Graph loaded: nv = ", nv(g), " ne = ", ne(g))
println("Running kadabra with 16 threads...")
println("Threads.nthreads() = ", Threads.nthreads())

# We will modify kadabra.jl to print progress!
