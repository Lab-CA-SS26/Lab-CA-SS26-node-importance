using Pkg
Pkg.activate(".")
using Graphs, StaticGraphs, Random

include("src/kadabra.jl")

println("Threads: $(Threads.nthreads())")

# Use Int32 like the real graph to avoid UInt8 issues  
dummy_g_raw = SimpleGraph{Int32}(3)
add_edge!(dummy_g_raw, 1, 2)
add_edge!(dummy_g_raw, 2, 3)
dummy_g = StaticGraph(dummy_g_raw)

println("dummy_g: nv=$(nv(dummy_g)), ne=$(ne(dummy_g)), eltype=$(eltype(dummy_g))")
flush(stdout)

# Test 1: sequential (should work)
println("\nTest 1: sequential on dummy graph...")
flush(stdout)
t1 = time()
res1 = kadabra_centrality(dummy_g, 0, 0.01, 0.1; start_factor=10, endpoints=false, parallel=false)
println("Sequential done in $(round(time()-t1, digits=2))s, n_samples=$(res1.n_samples)")
flush(stdout)

# Test 2: parallel 
println("\nTest 2: parallel on dummy graph...")
flush(stdout)
t2 = time()
res2 = kadabra_centrality(dummy_g, 0, 0.01, 0.1; start_factor=10, endpoints=false, parallel=true)
println("Parallel done in $(round(time()-t2, digits=2))s, n_samples=$(res2.n_samples)")
flush(stdout)
