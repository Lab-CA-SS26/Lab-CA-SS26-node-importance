using Pkg
Pkg.activate(joinpath(@__DIR__))
using Graphs
using DelimitedFiles
using DataFrames

include("src/kadabra.jl")

function run_local_test()
    datasets = [
        ("datasets/p2p-Gnutella31.txt", false),
        ("datasets/soc-Epinions1.txt", true)
    ]
    
    results = []
    
    # Warmup
    println("Warming up KADABRA...")
    g_warmup = cycle_graph(100)
    kadabra_centrality(g_warmup, 0, 0.1, 0.1)
    
    for (file, is_directed) in datasets
        println("\nLoading ", basename(file), "...")
        edges = readdlm(file, '\t', Int, comments=true, comment_char='#')
        n = maximum(edges) + 1
        g = is_directed ? SimpleDiGraph(n) : SimpleGraph(n)
        for i in 1:size(edges, 1)
            add_edge!(g, edges[i, 1] + 1, edges[i, 2] + 1)
        end
        N = nv(g)
        M = ne(g)
        println("Loaded. Nodes: $N, Edges: $M")
        
        println("Running Julia KADABRA...")
        allocs = @allocated begin
            t_jl = @elapsed kadabra_centrality(g, 0, 0.01, 0.01)
        end
        
        mem_mb = allocs / 1024^2
        
        push!(results, (basename(file), is_directed, N, M, "KADABRA (Julia)", t_jl, mem_mb))
        
        println("Mocking C++ KADABRA (Simulated Time)...")
        # Simulate C++ based on average server ratio
        t_cpp = t_jl * 0.45 
        push!(results, (basename(file), is_directed, N, M, "KADABRA (C++)", t_cpp, 0.0))
    end
    
    df = DataFrame(results, [:Dataset, :Directed, :Nodes, :Edges, :Method, :WallClock_s, :Memory_MB])
    println("\n=== Local Benchmark Results ===")
    display(df)
    println()
end

run_local_test()
