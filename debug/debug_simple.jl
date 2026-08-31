using Pkg
Pkg.activate(".")
using Graphs, Random

include("src/kadabra.jl")

println("Threads: $(Threads.nthreads())")

# Plain SimpleGraph, no StaticGraph at all
g = SimpleGraph{Int64}(3)
add_edge!(g, 1, 2)
add_edge!(g, 2, 3)

println("g: nv=$(nv(g)), ne=$(ne(g)), eltype=$(eltype(g)), type=$(typeof(g))")
flush(stdout)

# Test: parallel on SimpleGraph{Int64}
println("\nRunning parallel kadabra on SimpleGraph{Int64}(3) with $(Threads.nthreads()) threads...")
flush(stdout)
t1 = time()
res = kadabra_centrality(g, 0, 0.01, 0.1; start_factor=10, endpoints=false, parallel=true)
println("Done in $(round(time()-t1, digits=2))s, n_samples=$(res.n_samples)")
flush(stdout)
