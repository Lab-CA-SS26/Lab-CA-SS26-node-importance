using Pkg
Pkg.activate(".")
using Graphs, StaticGraphs, Random

include("src/kadabra.jl")

dummy_g_raw = SimpleGraph{Int64}(3)
add_edge!(dummy_g_raw, 1, 2)
add_edge!(dummy_g_raw, 2, 3)
dummy_g = StaticGraph(dummy_g_raw)

println("Running 1 thread loop...")
for i in 1:1000
    kadabra_centrality(dummy_g, 0, 0.01, 0.1; start_factor=10, endpoints=false)
end
println("Finished successfully!")
