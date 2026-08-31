using Pkg
Pkg.activate(".")
using Graphs, StaticGraphs, Random

include("src/kadabra.jl")

open("hang_log.txt", "w") do io
    println(io, "Starting...")
    flush(io)
    dummy_g_raw = SimpleGraph{Int64}(3)
    add_edge!(dummy_g_raw, 1, 2)
    add_edge!(dummy_g_raw, 2, 3)
    dummy_g = StaticGraph(dummy_g_raw)

    println(io, "Calling kadabra_centrality...")
    flush(io)
    
    # Redirect stdout to file to catch all prints
    redirect_stdout(io) do
        res = kadabra_centrality(dummy_g, 0, 0.01, 0.1; start_factor=10, endpoints=false)
        println("Done! ", res)
    end
end
