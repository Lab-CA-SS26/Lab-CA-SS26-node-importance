using Pkg
Pkg.activate(".")
using Graphs, StaticGraphs, Random

include("debug_kadabra_test.jl")

println("Starting...")
dummy_g_raw = SimpleGraph{Int64}(3)
add_edge!(dummy_g_raw, 1, 2)
add_edge!(dummy_g_raw, 2, 3)
dummy_g = StaticGraph(dummy_g_raw)

println("Calling kadabra_centrality...")
res = kadabra_centrality(dummy_g, 0, 0.01, 0.1; start_factor=10, endpoints=false)
println("Done! ", res)
